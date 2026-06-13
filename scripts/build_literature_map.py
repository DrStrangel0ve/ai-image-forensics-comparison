from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LITERATURE = [
    {
        "key": "universal_fake_detectors_2023",
        "title": "Towards Universal Fake Image Detectors that Generalize Across Generative Models",
        "year": 2023,
        "primary_url": "https://arxiv.org/abs/2302.10174",
        "theme": "foundation generalization",
        "use_in_paper": "Related work; motivation for frozen CLIP-style probes and source-heldout transfer.",
        "supports_claim": "Closed-set detector accuracy can fail under unseen generators; large pretrained features can generalize better.",
        "caveat": "Use as motivation for frozen encoders, not as proof that CLIP always wins.",
    },
    {
        "key": "genimage_2023",
        "title": "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image",
        "year": 2023,
        "primary_url": "https://arxiv.org/abs/2306.08571",
        "theme": "cross-generator benchmark",
        "use_in_paper": "Related work and dataset discussion for cross-generator/degraded-image evaluation.",
        "supports_claim": "Generator diversity and degraded-image testing should be first-class evaluation axes.",
        "caveat": "Do not imply the current repo has run the full GenImage benchmark yet.",
    },
    {
        "key": "aide_chameleon_2025",
        "title": "A Sanity Check for AI-generated Image Detection",
        "year": 2025,
        "primary_url": "https://openreview.net/forum?id=ODRHZrkOQM",
        "theme": "multi-expert detection",
        "use_in_paper": "Related work for heterogeneous expert branches and Chameleon-style stress tests.",
        "supports_claim": "Multi-expert CLIP plus frequency/artifact signals are a plausible answer to source shift.",
        "caveat": "Frame SCP-Fusion as a diagnostic protocol, not as a direct AIDE reproduction.",
    },
    {
        "key": "realhd_2026",
        "title": "RealHD",
        "year": 2026,
        "primary_url": "https://arxiv.org/abs/2602.10546",
        "theme": "recent dataset and residual noise",
        "use_in_paper": "Dataset/benchmark discussion and motivation for richer residual-noise features.",
        "supports_claim": "Recent generators and metadata-rich datasets make source-aware validation necessary.",
        "caveat": "Keep RealHD as a future benchmark unless it is actually downloaded and audited.",
    },
    {
        "key": "dire_2023",
        "title": "DIRE for Diffusion-Generated Image Detection",
        "year": 2023,
        "primary_url": "https://arxiv.org/abs/2303.09295",
        "theme": "diffusion reconstruction error",
        "use_in_paper": "Related work for reconstruction-error detection and future AEROBLADE-lite branch.",
        "supports_claim": "Generated and real images can differ in reconstruction behavior under diffusion priors.",
        "caveat": "DIRE is heavier than the current implementation; describe reconstruction features as roadmap unless run.",
    },
    {
        "key": "aeroblade_2024",
        "title": "AEROBLADE",
        "year": 2024,
        "primary_url": "https://arxiv.org/abs/2401.17879",
        "theme": "autoencoder reconstruction",
        "use_in_paper": "Methods roadmap for AEROBLADE-lite reconstruction residual features.",
        "supports_claim": "Latent-autoencoder reconstruction error is a practical physics-adjacent signal.",
        "caveat": "Current combined_v4 only includes lightweight reconstruction-style residual summaries.",
    },
    {
        "key": "fire_2025",
        "title": "FIRE",
        "year": 2025,
        "primary_url": "https://arxiv.org/abs/2412.07140",
        "theme": "frequency reconstruction",
        "use_in_paper": "Related work for band-limited reconstruction and robustness to perturbations.",
        "supports_claim": "Reconstruction error and frequency decomposition should be tested together.",
        "caveat": "Do not cite FIRE as implemented unless a true band-limited reconstruction branch is added.",
    },
    {
        "key": "spectral_any_resolution_2025",
        "title": "Any-Resolution AI-Generated Image Detection by Spectral Learning",
        "year": 2025,
        "primary_url": "https://arxiv.org/abs/2411.19417",
        "theme": "spectral learning",
        "use_in_paper": "Related work for spectral features and native/tiled evaluation.",
        "supports_claim": "Real-image spectral structure may be more stable than generator-specific artifacts.",
        "caveat": "The repo currently uses scalar/tiling diagnostics, not a learned spectral transformer.",
    },
    {
        "key": "no_pixel_left_behind_2025",
        "title": "No Pixel Left Behind",
        "year": 2025,
        "primary_url": "https://arxiv.org/abs/2508.17346",
        "theme": "high-resolution tiling",
        "use_in_paper": "Motivation for native tiling and avoiding over-reliance on resized 128x128 inputs.",
        "supports_claim": "High-resolution/local evidence can be lost by aggressive resizing.",
        "caveat": "Current native-tiling result is bounded to the conventional target branch.",
    },
    {
        "key": "fake_or_jpeg_2024",
        "title": "Fake or JPEG?",
        "year": 2024,
        "primary_url": "https://arxiv.org/abs/2403.17608",
        "theme": "compression bias",
        "use_in_paper": "Bias and robustness warning for JPEG/resize stress tests and dataset auditing.",
        "supports_claim": "Detectors can learn compression or size shortcuts instead of generation evidence.",
        "caveat": "Always pair JPEG findings with source-heldout and robustness context.",
    },
    {
        "key": "bias_free_training_2025",
        "title": "A Bias-Free Training Paradigm for More General AI-generated Image Detection",
        "year": 2025,
        "primary_url": "https://arxiv.org/html/2412.17671v2",
        "theme": "semantic bias",
        "use_in_paper": "Related work for dataset bias and semantically aligned real/fake pairs.",
        "supports_claim": "Content/semantic bias can confound detector training and evaluation.",
        "caveat": "Use as a future training direction unless aligned-pair data is added here.",
    },
    {
        "key": "photometric_faces_2023",
        "title": "A Geometric and Photometric Exploration of GAN and Diffusion Synthesized Faces",
        "year": 2023,
        "primary_url": "https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Bohacek_A_Geometric_and_Photometric_Exploration_of_GAN_and_Diffusion_Synthesized_CVPRW_2023_paper.html",
        "theme": "physics-informed analysis",
        "use_in_paper": "Related work for photometric/geometric cues in generated-image forensics.",
        "supports_claim": "Physical inconsistency can be a useful forensic signal even when local texture is plausible.",
        "caveat": "Face-focused analysis; the current repo uses generic single-image proxy features.",
    },
    {
        "key": "light2lie_2026",
        "title": "Light2Lie: Detecting Deepfake Images Using Physical Reflectance Laws",
        "year": 2026,
        "primary_url": "https://www.ndss-symposium.org/ndss-paper/light2lie-detecting-deepfake-images-using-physical-reflectance-laws/",
        "theme": "reflectance physics",
        "use_in_paper": "Motivation for upgrading pseudo-normal features toward reflectance/specular consistency.",
        "supports_claim": "Physical reflectance laws can expose light-surface inconsistencies in synthetic imagery.",
        "caveat": "Do not claim this repo implements Light2Lie-style reflectance estimation yet.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-facing literature map for WIFS/DFF drafting.")
    parser.add_argument("--out-csv", default="reports/assets/literature_map.csv")
    parser.add_argument("--out-md", default="reports/literature_map_2026_06_14.md")
    return parser.parse_args()


def build_literature_map() -> pd.DataFrame:
    frame = pd.DataFrame(LITERATURE)
    expected = {"key", "title", "year", "primary_url", "theme", "use_in_paper", "supports_claim", "caveat"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Literature map is missing columns: {sorted(missing)}")
    if frame["key"].duplicated().any():
        duplicates = frame.loc[frame["key"].duplicated(), "key"].tolist()
        raise ValueError(f"Duplicate literature keys: {duplicates}")
    return frame


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(frame: pd.DataFrame, out_md: Path) -> None:
    theme_counts = frame.groupby("theme").size().reset_index(name="count")
    lines = [
        "# Literature Map",
        "",
        "Run date: 2026-06-14",
        "",
        "Generated by `scripts/build_literature_map.py` as a paper-drafting bridge between the research deep dive and the WIFS/DFF skeletons.",
        "",
        "This is not a finished bibliography. It is a claim-to-literature planning map: each row says where the work belongs in the paper and what caveat should survive editing.",
        "",
        "## Theme Counts",
        "",
        _markdown_table(theme_counts, ["theme", "count"]),
        "",
        "## Paper-Facing Map",
        "",
        _markdown_table(
            frame,
            ["key", "title", "year", "theme", "use_in_paper", "supports_claim", "caveat", "primary_url"],
        ),
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = build_literature_map()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    write_markdown(frame, Path(args.out_md))
    print(out_csv.resolve())
    print(Path(args.out_md).resolve())


if __name__ == "__main__":
    main()
