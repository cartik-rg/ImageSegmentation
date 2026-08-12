from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_processing.image_segmenter import run_model_comparison

DEFAULT_MODELS = [
    "sam2",
    "groundingdino_sam2",
    "mask2former",
    "yolo_seg",
    "anomalib_patchcore",
]

DEFAULT_PROMPT = (
    "corroded area . cracked pane . missing rivet . damaged wire . broken panel . "
    "wing . cockpit . fuselage . wheel assembly . tail . flap . aileron . engine . propeller . landing gear"
)


def _summarize_metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in results:
        metrics = item.get("metrics", {})
        latency = float(metrics.get("inference_latency_ms", 0.0))
        grouped[item["model_name"]].append(latency)

    summary_rows: list[dict[str, Any]] = []
    for model_name, values in sorted(grouped.items()):
        summary_rows.append({
            "model": model_name,
            "image_count": len(values),
            "avg_inference_latency_ms": round(sum(values) / len(values), 3),
            "min_inference_latency_ms": round(min(values), 3),
            "max_inference_latency_ms": round(max(values), 3),
        })

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all requested segmentation backends across every image in the input folder"
    )
    parser.add_argument(
        "image_dir",
        default="input/image",
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Vocabulary prompt used to guide object recognition",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root output folder used for segmented_images, segmented_image_triples, and performance_metrics",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="One or more model backends to execute",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    performance_dir = output_dir / "performance_metrics"
    performance_dir.mkdir(parents=True, exist_ok=True)

    results = run_model_comparison(
        image_dir=image_dir,
        prompt=args.prompt,
        models=args.models,
        output_dir=output_dir,
    )

    performance_summary = _summarize_metrics(results)
    summary_path = performance_dir / "model_comparison_summary.json"
    summary_path.write_text(json.dumps(performance_summary, indent=2), encoding="utf-8")

    per_model_dir = performance_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        model_name = result["model_name"]
        metrics = result.get("metrics", {})
        model_metrics_path = per_model_dir / f"{model_name}_metrics.json"
        model_metrics_path.write_text(
            json.dumps(
                {
                    "image_path": result["image_path"],
                    "segmented_image_path": result["segmented_image_path"],
                    "triples_path": result["triples_path"],
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Comparison results written to: {output_dir}")
    print(f"Performance summary written to: {summary_path}")


if __name__ == "__main__":
    main()
