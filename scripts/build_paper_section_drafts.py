from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from source_stress_utils import load_source_stress_summary
from tiled_dino_tradeoff_utils import load_tiled_dino_tradeoff_summary


REQUIRED_FINDINGS = [
    "ishu_same_combined_v3",
    "ishu_same_resnet18",
    "ishu_same_physics_guided",
    "ishu_to_ms_clip_standalone",
    "ishu_to_ms_scp_fusion_all_foundation",
    "ishu_to_ms_triage5_clip_standalone",
    "ms_to_ishu_tuned_fusion_constraint_sweep_best",
    "ms_to_ishu_tuned_fusion_native_tiling_best",
    "ms_to_ishu_tuned_fusion_jpeg30",
    "ms_to_ishu_tuned_fusion_blur1",
    "ms_to_ishu_tuned_fusion_social_720p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WIFS/DFF manuscript section drafts from checked-in result and literature assets."
    )
    parser.add_argument("--core-results", default="reports/assets/publication_core_results.csv")
    parser.add_argument("--claim-matrix", default="reports/assets/claim_evidence_matrix.csv")
    parser.add_argument("--literature-map", default="reports/assets/literature_map.csv")
    parser.add_argument("--tiled-dino-tradeoff", default="reports/assets/tiled_dinov2_calibration_tradeoff.csv")
    parser.add_argument(
        "--source-stress-summary",
        default="reports/assets/ms_cocoai_to_ishu_source_holdout_model_selection_source_summary.csv",
    )
    parser.add_argument("--out-path", default="reports/paper_section_drafts_2026_06_14.md")
    parser.add_argument("--manifest-out", default="reports/assets/paper_section_draft_manifest.csv")
    return parser.parse_args()


def _rows_by_id(core: pd.DataFrame) -> dict[str, pd.Series]:
    rows = {}
    for finding_id in REQUIRED_FINDINGS:
        matches = core[core["finding_id"] == finding_id]
        if matches.empty:
            raise ValueError(f"Missing required finding_id={finding_id!r}")
        rows[finding_id] = matches.iloc[0]
    return rows


def _fmt(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _section(title: str, text: str) -> dict[str, object]:
    return {"section": title, "text": text.strip(), "word_count": _word_count(text)}


def _citation_keys(literature: pd.DataFrame, theme: str) -> str:
    keys = literature.loc[literature["theme"].eq(theme), "key"].astype(str).tolist()
    if not keys:
        raise ValueError(f"Missing literature theme={theme!r}")
    return ", ".join(f"`{key}`" for key in keys)


def build_sections(
    core_results: Path,
    claim_matrix: Path,
    literature_map: Path,
    tiled_dino_tradeoff: Path,
    source_stress_summary: Path,
) -> list[dict[str, object]]:
    core = pd.read_csv(core_results)
    claims = pd.read_csv(claim_matrix)
    literature = pd.read_csv(literature_map)
    tiled_dino = load_tiled_dino_tradeoff_summary(tiled_dino_tradeoff)
    if tiled_dino is None:
        raise ValueError("Missing tiled-DINO tradeoff summary")
    source_stress = load_source_stress_summary(source_stress_summary)
    if source_stress is None:
        raise ValueError("Missing source-stress summary")
    rows = _rows_by_id(core)

    combined_v3 = rows["ishu_same_combined_v3"]
    resnet = rows["ishu_same_resnet18"]
    physics = rows["ishu_same_physics_guided"]
    clip = rows["ishu_to_ms_clip_standalone"]
    fusion = rows["ishu_to_ms_scp_fusion_all_foundation"]
    triage = rows["ishu_to_ms_triage5_clip_standalone"]
    reverse = rows["ms_to_ishu_tuned_fusion_constraint_sweep_best"]
    native = rows["ms_to_ishu_tuned_fusion_native_tiling_best"]
    jpeg30 = rows["ms_to_ishu_tuned_fusion_jpeg30"]
    blur = rows["ms_to_ishu_tuned_fusion_blur1"]
    social = rows["ms_to_ishu_tuned_fusion_social_720p"]

    ready_claims = claims[claims["status"].isin(["ready", "ready_with_caveat"])]
    claim_count = len(ready_claims)
    sections = [
        _section(
            "WIFS Introduction Draft",
            (
                "AI-generated image detection is often summarized by a single accuracy number on a familiar "
                "benchmark, but that framing hides the decision problem faced by forensic reviewers. A useful "
                "detector must keep ranking, calibration, threshold behavior, and high-confidence triage stable "
                "when the generator source or image processing pipeline changes. This project therefore treats "
                "real-vs-generated image detection as a source-heldout evaluation problem. The current evidence "
                f"starts with a same-domain anchor where `combined_v3` and ResNet-18 both reach "
                f"{_fmt(combined_v3, 'accuracy')} accuracy on Ishu, while physics-guided ResNet fusion improves "
                f"to {_fmt(physics, 'accuracy')} accuracy and {_fmt(physics, 'auc')} AUC. The transfer results "
                "then separate ranking strength from deployable decisions: frozen CLIP is the strongest Ishu to "
                f"MS COCOAI ranker at {_fmt(clip, 'auc')} AUC, while source-aware reverse tuned fusion gives "
                f"a more usable operating point at {_fmt(reverse, 'accuracy')} accuracy with a "
                f"{_fmt(reverse, 'predicted_fake_rate')} target fake-call rate."
            ),
        ),
        _section(
            "WIFS Related Work Draft",
            (
                "Prior work motivates three design choices in this benchmark. First, universal fake-detector and "
                "cross-generator benchmark studies motivate frozen foundation encoders and source-heldout testing "
                f"({ _citation_keys(literature, 'foundation generalization') }, "
                f"{ _citation_keys(literature, 'cross-generator benchmark') }). Second, multi-expert and spectral "
                "detectors motivate keeping physical/signal, frequency, reconstruction, and foundation branches "
                f"separate enough to inspect ({ _citation_keys(literature, 'multi-expert detection') }, "
                f"{ _citation_keys(literature, 'spectral learning') }). Third, compression-bias and high-resolution "
                "work warn against over-interpreting gains that disappear under JPEG, resize, crop, or tiling "
                f"changes ({ _citation_keys(literature, 'compression bias') }, "
                f"{ _citation_keys(literature, 'high-resolution tiling') }). The physics branch is framed with "
                "the same caution: photometric and reflectance-based forensics motivate physical cues, but this "
                "repo currently uses a single-image physical/signal proxy rather than calibrated multi-light "
                f"photometric stereo ({ _citation_keys(literature, 'physics-informed analysis') }, "
                f"{ _citation_keys(literature, 'reflectance physics') })."
            ),
        ),
        _section(
            "WIFS Data And Audit Draft",
            (
                "The current submission package centers on two audited evaluation directions. Ishu AI-vs-real "
                "provides repeated same-domain splits for comparing `combined_v3`, ResNet-18, physics-guided "
                "fusion, frozen encoders, and saved-score fusion. Source-balanced MS COCOAI provides generator "
                "source labels for cross-domain and source-aware validation with SD3, SD2.1, SDXL, DALL-E 3, and "
                "MidJourney-style generated subsets represented in the exported metadata. Dataset commands, "
                "audits, and artifact manifests are part of the public repo so that results can be traced back to "
                "specific exports rather than opaque benchmark names. The paper should state that raw datasets and "
                "large model artifacts are external, while reports, prediction-derived tables, figures, manifests, "
                "and reproduction commands are checked in. The source-holdout stress table adds a generator-level "
                f"audit: for the paper-facing capped policy, `{source_stress['heldout_source_name']}` is currently "
                f"the weakest held-out generator, with {source_stress['recall']} recall and "
                f"{source_stress['fake_miss_rate']} fake-miss rate."
            ),
        ),
        _section(
            "WIFS Methods Draft",
            (
                "The compared methods are intentionally heterogeneous. `combined_v3` measures single-image "
                "physical/signal features such as pseudo-normal consistency, noise residuals, JPEG/block cues, "
                "frequency balance, and chroma consistency. ResNet-18 is the conventional neural baseline. The "
                "physics-guided model fuses a ResNet image embedding with standardized `combined_v3` features, "
                "which is the practical physics-informed route for single-image datasets. Frozen ConvNeXt, DINOv2, "
                "and CLIP encoders test whether broad pretrained representations transfer better than task-trained "
                "detectors. SCP-Fusion combines saved branch scores and evaluates source-aware calibration, "
                "branch constraints, and threshold policies, reporting AUC, accuracy, Brier score, ECE, fake-call "
                "rate, and partial triage coverage instead of a single headline metric."
            ),
        ),
        _section(
            "WIFS Results Draft",
            (
                f"Same-domain Ishu results show why the physical branch is worth keeping: physics-guided fusion "
                f"improves from the tied `combined_v3`/ResNet-18 accuracy of {_fmt(combined_v3, 'accuracy')} to "
                f"{_fmt(physics, 'accuracy')} accuracy and {_fmt(physics, 'auc')} AUC. Cross-domain transfer changes "
                f"the ranking: frozen CLIP reaches {_fmt(clip, 'accuracy')} accuracy and {_fmt(clip, 'auc')} AUC on "
                f"Ishu to source-balanced MS COCOAI, ahead of all-foundation SCP-Fusion at "
                f"{_fmt(fusion, 'accuracy')} accuracy and {_fmt(fusion, 'auc')} AUC. The triage result is more "
                f"operationally useful than a forced threshold: CLIP decides {_fmt(triage, 'coverage')} of target "
                f"images at {_fmt(triage, 'decided_accuracy')} decided-case accuracy. In the reverse direction, "
                f"source-capped tuned fusion reaches {_fmt(reverse, 'accuracy')} accuracy and "
                f"{_fmt(reverse, 'auc')} AUC, and the bounded native-tiled conventional branch diagnostic improves "
                f"that to {_fmt(native, 'accuracy')} accuracy and {_fmt(native, 'auc')} AUC. Robustness remains mixed: "
                f"social-style 720p processing is comparatively stable at {_fmt(social, 'accuracy')} accuracy and "
                f"{_fmt(social, 'auc')} AUC, while JPEG30 and blur expose weaker operating points at "
                f"{_fmt(jpeg30, 'accuracy')} and {_fmt(blur, 'accuracy')} accuracy. Held-out-generator stress "
                f"adds a source-specific failure handle: `{source_stress['heldout_source_name']}` is the weakest "
                f"capped source-holdout generator at {source_stress['recall']} recall and "
                f"{source_stress['fake_miss_rate']} fake-miss rate. A tiled-DINO follow-up gives "
                f"`tile_max` average deltas of {tiled_dino['tile_max_acc_delta']} accuracy and "
                f"{tiled_dino['tile_max_auc_delta']} AUC across {tiled_dino['n_transforms']} stress probes, while "
                f"`tile_mean` improves Brier on {tiled_dino['tile_mean_brier_count']} and ECE on "
                f"{tiled_dino['tile_mean_ece_count']} probes."
            ),
        ),
        _section(
            "DFF Expansion Draft",
            (
                "The DFF version should make SCP-Fusion the diagnostic protocol and organizing frame rather than a single model claim. "
                "The strongest story is that different branches answer different forensic questions: CLIP gives the "
                "best standalone transfer ranking, physics-guided fusion improves the same-domain and some "
                "calibration anchors, source-capped fusion makes reverse binary decisions less biased, and strict "
                "two-threshold triage avoids pretending every image deserves a confident binary call. Failure grids "
                "and source-slice diagnostics should be used as explainability evidence: when generated images are "
                "missed, the paper should ask whether the miss is semantic, spectral, compression-driven, or a "
                "source-threshold artifact. The DFF framing can carry the tiled-DINO mode tradeoff as a small "
                "robustness design rule, then connect it to the `combined_v4` and reconstruction roadmap; it should "
                "keep the caveat that current `combined_v4` is an ablation candidate and that true "
                "AEROBLADE/FIRE-style reconstruction has not yet replaced the lightweight residual branch."
            ),
        ),
        _section(
            "Limitations And Reproducibility Draft",
            (
                f"The manuscript should preserve all {claim_count} ready or caveated claim guardrails from the "
                "claim-evidence matrix. Most importantly, the physical branch is a single-image proxy rather than "
                "classic photometric stereo, SCP-Fusion does not universally beat frozen CLIP, native/foundation "
                "tiling is a bounded diagnostic rather than an official external high-resolution benchmark result, "
                "and robustness claims must name the transform being tested. The repo is public and contains generated result tables, "
                "LaTeX fragments, paper skeletons, lint reports, literature maps, draft BibTeX, and reproduction "
                "commands. The final paper still needs official venue templates, verified bibliography metadata, "
                "and, if time allows, broader source-balanced data or a true reconstruction branch."
            ),
        ),
    ]
    return sections


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("\n", " ").replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(sections: list[dict[str, object]], out_path: Path) -> pd.DataFrame:
    manifest = pd.DataFrame(
        [
            {
                "section": section["section"],
                "word_count": section["word_count"],
                "has_metric": bool(re.search(r"\d\.\d{4}", str(section["text"]))),
                "has_caveat": "single-image" in str(section["text"]) or "does not universally beat" in str(section["text"]),
            }
            for section in sections
        ]
    )
    lines = [
        "# Paper Section Drafts",
        "",
        "Run date: 2026-06-14",
        "",
        "Generated by `scripts/build_paper_section_drafts.py` from the publication core results, claim-evidence matrix, and literature map.",
        "",
        "These are prose starting points for WIFS/DFF authoring. They are not final copy and still need venue-template editing, bibliography metadata verification, and coauthor review.",
        "",
        "## Section Manifest",
        "",
        _markdown_table(manifest),
        "",
    ]
    for section in sections:
        lines.extend([f"## {section['section']}", "", str(section["text"]), ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    sections = build_sections(
        Path(args.core_results),
        Path(args.claim_matrix),
        Path(args.literature_map),
        Path(args.tiled_dino_tradeoff),
        Path(args.source_stress_summary),
    )
    manifest = write_report(sections, Path(args.out_path))
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)
    print(Path(args.out_path).resolve())
    print(manifest_out.resolve())


if __name__ == "__main__":
    main()
