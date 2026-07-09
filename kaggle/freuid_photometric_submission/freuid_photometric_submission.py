from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


COMPETITION_SLUG = "the-freuid-challenge-2026-ijcai-ecai"
WORK_DIR = Path("/kaggle/working")
INPUT_CANDIDATES = [
    Path("/kaggle/input/competitions") / COMPETITION_SLUG,
    Path("/kaggle/input") / COMPETITION_SLUG,
]
IMAGE_SIZE = 128
MAX_TRAIN = 640
MAX_VAL = 160
SEED = 7

FEATURE_NAMES = [
    "gray_mean",
    "gray_std",
    "gray_skew",
    "edge_mean",
    "edge_std",
    "edge_p90",
    "edge_p99",
    "normal_z_mean",
    "normal_z_std",
    "normal_z_p05",
    "normal_xy_std",
    "slope_mean",
    "slope_std",
    "integrability_abs_mean",
    "integrability_std",
    "integrability_p95",
    "laplacian_abs_mean",
    "laplacian_std",
    "high_frequency_std",
    "saturation_mean",
    "saturation_std",
    "edge_saturation_corr",
    "rg_corr",
    "rb_corr",
    "gb_corr",
    "normal_z_entropy",
]


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_path_score(value: object, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def describe_input_tree() -> None:
    print("kaggle_input_tree", flush=True)
    for source in sorted(Path("/kaggle/input").glob("*")):
        print(f"- {source}", flush=True)
        try:
            for child in sorted(source.iterdir())[:50]:
                print(f"  - {child.name}", flush=True)
        except NotADirectoryError:
            pass


def find_input_file(filename: str) -> Path:
    direct = []
    for root in INPUT_CANDIDATES:
        direct.extend([root / filename, root / "small_files" / filename])
    for candidate in direct:
        if candidate.exists():
            return candidate
    matches = sorted(Path("/kaggle/input").glob(f"**/{filename}"), key=lambda path: (len(path.parts), str(path)))
    if matches:
        return matches[0]
    describe_input_tree()
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input")


def infer_image_root(train_labels: Path) -> Path:
    candidates = [train_labels.parent, *train_labels.parents]
    candidates.extend(INPUT_CANDIDATES)
    candidates.extend(sorted(Path("/kaggle/input").glob("*")))
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "public_test").exists():
            return candidate
    describe_input_tree()
    raise FileNotFoundError("Could not infer FREUID image root containing train/ and public_test/")


def freuid_competition_path(value: object, split: str | None = None) -> str:
    raw = str(value).replace("\\", "/").strip()
    parts = PurePosixPath(raw).parts
    name = parts[-1]
    if "." not in name:
        name = f"{name}.jpeg"
    inferred_split = split
    if inferred_split is None and parts and parts[0] in {"train", "public_test"}:
        inferred_split = parts[0]
    inferred_split = inferred_split or "public_test"
    return f"{inferred_split}/{inferred_split}/{name}"


def resolve_image_path(image_root: Path, image_path: object) -> Path:
    raw = str(image_path)
    candidates = [
        image_root / freuid_competition_path(raw),
        image_root / raw.replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for {raw}; tried {[str(path) for path in candidates]}")


def split_labels(labels: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    labels["_score"] = labels["id"].astype(str).map(lambda value: stable_path_score(value, SEED))
    train_indices: list[int] = []
    val_indices: list[int] = []
    for _key, group in labels.groupby(["type", "label"], sort=True):
        ordered = group.sort_values(["_score", "id"], kind="mergesort")
        n_val = min(len(ordered) - 1, max(1, int(round(len(ordered) * val_fraction))))
        val_indices.extend(ordered.index[:n_val].tolist())
        train_indices.extend(ordered.index[n_val:].tolist())
    train = labels.loc[train_indices].drop(columns=["_score"]).reset_index(drop=True)
    val = labels.loc[val_indices].drop(columns=["_score"]).reset_index(drop=True)
    return train, val


def balanced_limit(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame.reset_index(drop=True)
    scored = frame.copy()
    scored["_score"] = scored["id"].astype(str).map(lambda value: stable_path_score(value, seed))
    groups = [group.sort_values(["_score", "id"], kind="mergesort") for _key, group in scored.groupby(["type", "label"], sort=True)]
    selected: list[int] = []
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for group in groups:
            if cursor < len(group):
                selected.append(int(group.index[cursor]))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return scored.loc[selected].drop(columns=["_score"]).reset_index(drop=True)


def safe_skew(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    std = float(flat.std())
    if std < 1e-8:
        return 0.0
    return float(np.mean(((flat - flat.mean()) / std) ** 3))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if float(a_flat.std()) < 1e-8 or float(b_flat.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def box_blur_3x3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="reflect")
    acc = np.zeros_like(values)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return acc / 9.0


def extract_photometric_features(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx**2 + gy**2)
    nx = -gx * 8.0
    ny = -gy * 8.0
    nz = np.ones_like(gray, dtype=np.float32)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / np.clip(norm, 1e-8, None)
    ny = ny / np.clip(norm, 1e-8, None)
    nz = nz / np.clip(norm, 1e-8, None)
    p = -nx / np.clip(nz, 1e-6, None)
    q = -ny / np.clip(nz, 1e-6, None)
    integrability = np.gradient(p, axis=0) - np.gradient(q, axis=1)
    laplacian = -4.0 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
    high_frequency = gray - box_blur_3x3(gray)
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    saturation = (max_rgb - min_rgb) / np.clip(max_rgb, 1e-6, None)
    nz_hist, _ = np.histogram(nz, bins=16, range=(0.0, 1.0), density=False)
    nz_probs = nz_hist.astype(np.float64) / max(1, nz_hist.sum())
    values = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_skew": safe_skew(gray),
        "edge_mean": float(edge.mean()),
        "edge_std": float(edge.std()),
        "edge_p90": float(np.percentile(edge, 90)),
        "edge_p99": float(np.percentile(edge, 99)),
        "normal_z_mean": float(nz.mean()),
        "normal_z_std": float(nz.std()),
        "normal_z_p05": float(np.percentile(nz, 5)),
        "normal_xy_std": float(np.sqrt(nx**2 + ny**2).std()),
        "slope_mean": float(np.sqrt(p**2 + q**2).mean()),
        "slope_std": float(np.sqrt(p**2 + q**2).std()),
        "integrability_abs_mean": float(np.abs(integrability).mean()),
        "integrability_std": float(integrability.std()),
        "integrability_p95": float(np.percentile(np.abs(integrability), 95)),
        "laplacian_abs_mean": float(np.abs(laplacian).mean()),
        "laplacian_std": float(laplacian.std()),
        "high_frequency_std": float(high_frequency.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "edge_saturation_corr": safe_corr(edge, saturation),
        "rg_corr": safe_corr(rgb[:, :, 0], rgb[:, :, 1]),
        "rb_corr": safe_corr(rgb[:, :, 0], rgb[:, :, 2]),
        "gb_corr": safe_corr(rgb[:, :, 1], rgb[:, :, 2]),
        "normal_z_entropy": -float(np.sum(nz_probs * np.log2(np.clip(nz_probs, 1e-12, None)))),
    }
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_matrix(frame: pd.DataFrame, image_root: Path, labeled: bool, desc: str) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for row in tqdm(frame.to_dict("records"), desc=desc):
        path = resolve_image_path(image_root, row["image_path"])
        features.append(extract_photometric_features(path))
        ids.append(str(row["id"]))
        if labeled:
            labels.append(int(row["label"]))
    y = np.asarray(labels, dtype=int) if labeled else None
    return np.vstack(features), y, ids


def apcer_at_bpcer(y_true: np.ndarray, scores: np.ndarray, bpcer_target: float = 0.01) -> tuple[float, float, float]:
    bona = scores[y_true == 0]
    attack = scores[y_true == 1]
    candidates = np.unique(np.concatenate([scores, np.nextafter(scores, np.inf), [-np.inf, np.inf]]))
    best: tuple[float, float, float] | None = None
    for threshold in np.sort(candidates):
        bpcer = float(np.mean(bona >= threshold))
        if bpcer > bpcer_target:
            continue
        apcer = float(np.mean(attack < threshold))
        candidate = (float(threshold), bpcer, apcer)
        if best is None or apcer < best[2] or (apcer == best[2] and bpcer > best[1]):
            best = candidate
    if best is None:
        raise RuntimeError("No threshold satisfied BPCER target")
    return best


def audet_proxy(y_true: np.ndarray, scores: np.ndarray) -> float:
    bona = scores[y_true == 0]
    attack = scores[y_true == 1]
    candidates = np.unique(np.concatenate([scores, np.nextafter(scores, np.inf), [-np.inf, np.inf]]))
    points = []
    for threshold in np.sort(candidates):
        points.append((float(np.mean(bona >= threshold)), float(np.mean(attack < threshold))))
    points = sorted(set(points))
    if len(points) < 2:
        return 0.0
    bpcer = np.asarray([point[0] for point in points])
    apcer = np.asarray([point[1] for point in points])
    return float(np.sum(np.diff(bpcer) * (apcer[:-1] + apcer[1:]) * 0.5))


def main() -> None:
    train_labels_path = find_input_file("train_labels.csv")
    sample_submission_path = find_input_file("sample_submission.csv")
    image_root = infer_image_root(train_labels_path)
    print(f"train_labels={train_labels_path}", flush=True)
    print(f"sample_submission={sample_submission_path}", flush=True)
    print(f"image_root={image_root}", flush=True)

    labels = pd.read_csv(train_labels_path)
    train_frame, val_frame = split_labels(labels)
    train_frame = balanced_limit(train_frame, MAX_TRAIN, SEED)
    val_frame = balanced_limit(val_frame, MAX_VAL, SEED + 1)
    sample = pd.read_csv(sample_submission_path)
    test_frame = pd.DataFrame({"id": sample["id"].astype(str)})
    test_frame["image_path"] = test_frame["id"].map(lambda value: f"public_test/{value}.jpeg")

    x_train, y_train, _train_ids = extract_matrix(train_frame, image_root, labeled=True, desc="freuid/train")
    x_val, y_val, val_ids = extract_matrix(val_frame, image_root, labeled=True, desc="freuid/val")
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
        ]
    )
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    threshold, bpcer, apcer = apcer_at_bpcer(y_val, val_scores)
    val_labels = (val_scores >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, val_labels)),
        "roc_auc": float(roc_auc_score(y_val, val_scores)),
        "brier_score": float(brier_score_loss(y_val, val_scores)),
        "apcer_at_1pct_bpcer": float(apcer),
        "bpcer_at_operating_point": float(bpcer),
        "audet_proxy": float(audet_proxy(y_val, val_scores)),
        "threshold": float(threshold),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "max_train": MAX_TRAIN,
        "max_val": MAX_VAL,
        "feature_set": "photometric",
        "image_size": IMAGE_SIZE,
    }
    write_json(metrics, WORK_DIR / "photometric_metrics.json")
    pd.DataFrame({"id": val_ids, "y_true": y_val, "fraud_score": val_scores, "label": val_labels}).to_csv(
        WORK_DIR / "val_predictions.csv",
        index=False,
    )
    print("local_validation=" + json.dumps(metrics, sort_keys=True), flush=True)

    x_test, _y_test, test_ids = extract_matrix(test_frame, image_root, labeled=False, desc="freuid/public_test")
    test_scores = model.predict_proba(x_test)[:, 1]
    test_labels = (test_scores >= threshold).astype(int)
    predictions = pd.DataFrame({"id": test_ids, "fraud_score": test_scores, "label": test_labels})
    predictions.to_csv(WORK_DIR / "public_predictions.csv", index=False)

    submission = pd.DataFrame({"id": sample["id"].astype(str), "label": test_labels.astype(int)})
    if list(submission["id"]) != list(sample["id"].astype(str)):
        raise RuntimeError("Submission ID order does not match sample submission")
    if not set(submission["label"].unique()).issubset({0, 1}):
        raise RuntimeError("Submission labels are not binary")
    submission.to_csv(WORK_DIR / "submission.csv", index=False)
    lint = {
        "status": "pass",
        "n_rows": int(len(submission)),
        "label_counts": {str(key): int(value) for key, value in submission["label"].value_counts().sort_index().items()},
        "sample_submission_path": str(sample_submission_path),
        "submission_path": str(WORK_DIR / "submission.csv"),
        "metrics": metrics,
    }
    write_json(lint, WORK_DIR / "submission_lint.json")
    print("submission_ready=/kaggle/working/submission.csv", flush=True)
    print(json.dumps(lint, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
