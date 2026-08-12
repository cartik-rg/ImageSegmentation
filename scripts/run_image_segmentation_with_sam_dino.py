from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_processing.image_segmenter import DinoSamImageSegmenter, run_model_comparison

DEFAULT_PROMPT = (
    "corroded area . cracked pane . missing rivet . damaged wire . broken panel . \
    wing . cockpit . fuselage . wheel assembly . tail . flap . aileron . engine . propeller . landing gear"
    )

DEFAULT_MODELS = [
    "sam2",
    "groundingdino_sam2",
    "mask2former",
    "yolo_seg",
    "anomalib_patchcore",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run image segmentation and extract triples across one or more backends"
    )
    parser.add_argument("image_path", help="Path to a JPEG/PNG image or a directory containing images")
    parser.add_argument(
        "--prompt",
        default=os.getenv("IMAGE_SEGMENTATION_PROMPT", DEFAULT_PROMPT),
        help="Vocabulary prompt used to guide the segmentation and object recognition step",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "./output"),
        help="Directory where segmented_images and segmented_image_triples will be written",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="One or more model backends to execute in the comparison run",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    results = []
    if image_path.is_dir():
        results = run_model_comparison(
            image_dir=image_path,
            prompt=args.prompt,
            models=args.models,
            output_dir=args.output_dir,
        )
    else:
        for model_name in args.models:
            segmenter = DinoSamImageSegmenter(
                prompt=args.prompt,
                output_dir=args.output_dir,
                model_name=model_name,
            )
            result = segmenter.segment_and_recognize(image_path)
            results.append(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "model_comparison_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for result in results:
        print(f"Model: {result['model_name']}")
        print(f"Segmented image written to: {result['segmented_image_path']}")
        print(f"Triples written to: {result['triples_path']}")
        metrics = result.get("metrics", {})
        print(
            "Metrics: "
            f"latency_ms={metrics.get('inference_latency_ms', 0.0)}, "
            f"gpu_mb={metrics.get('gpu_usage_mb', 0.0)}"
        )

    print(f"Comparison summary written to: {summary_path}")


if __name__ == "__main__":
    main()
