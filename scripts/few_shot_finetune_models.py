from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_processing.image_segmenter import DinoSamImageSegmenter

DEFAULT_MODELS = [
    "sam2",
    "groundingdino_sam2",
    "mask2former",
    "yolo_seg",
    "anomalib_patchcore",
]

DEFAULT_PROMPT = (
    "damaged aircraft engine . engine damage . fan blade . crack . corrosion . "
    "broken blade . deformation . dent . aircraft propulsion system"
)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class AnnotationSample:
    image_path: Path
    annotations: list[dict[str, Any]]
    source_json: Path


def resolve_image_dir(data_dir: Path) -> Path:
    if any(data_dir.glob("*.*")) and any(p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES for p in data_dir.iterdir() if p.is_file()):
        return data_dir

    for candidate in ["images", "image", "imgs", "pictures"]:
        candidate_dir = data_dir / candidate
        if candidate_dir.exists() and candidate_dir.is_dir():
            return candidate_dir

    parent = data_dir.parent
    for candidate in ["images", "image", "imgs", "pictures"]:
        candidate_dir = parent / candidate
        if candidate_dir.exists() and candidate_dir.is_dir():
            return candidate_dir

    raise FileNotFoundError(
        f"Could not locate training images in '{data_dir}' or adjacent folders. "
        "Place images in the same folder or in an 'image' / 'images' subfolder."
    )


def load_annotation_samples(data_dir: Path) -> list[AnnotationSample]:
    data_dir = data_dir.resolve()
    annotation_dir = data_dir
    annotation_files = sorted(annotation_dir.glob("*.json"))
    if not annotation_files:
        nested = data_dir / "annotations"
        if nested.exists() and nested.is_dir():
            annotation_dir = nested
            annotation_files = sorted(annotation_dir.glob("*.json"))

    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotation files found in {data_dir} or {data_dir / 'annotations'}")

    image_dir = resolve_image_dir(annotation_dir)
    samples: list[AnnotationSample] = []

    for annotation_path in annotation_files:
        content = json.loads(annotation_path.read_text(encoding="utf-8"))
        file_name = content.get("file_name") or content.get("image_id")
        if not file_name:
            raise ValueError(f"Missing file_name in annotation file {annotation_path}")

        image_path = image_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file '{file_name}' referenced by {annotation_path.name} was not found in {image_dir}"
            )

        annotations = content.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"Invalid or empty annotations in {annotation_path}")

        samples.append(AnnotationSample(image_path=image_path, annotations=annotations, source_json=annotation_path))

    return samples


def normalize_bbox(bbox: dict[str, int], width: int, height: int) -> tuple[float, float, float, float]:
    x0 = bbox["x0"]
    y0 = bbox["y0"]
    x1 = bbox["x1"]
    y1 = bbox["y1"]
    xc = (x0 + x1) / 2.0 / width
    yc = (y0 + y1) / 2.0 / height
    w = (x1 - x0) / width
    h = (y1 - y0) / height
    return xc, yc, w, h


def build_yolo_dataset(samples: list[AnnotationSample], output_dir: Path) -> tuple[Path, list[str]]:
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The YOLO-Seg training pipeline requires the ultralytics package. "
            "Install it with `pip install ultralytics`." 
        ) from exc

    dataset_dir = output_dir / "yolo_dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_names = []
    for sample in samples:
        for annotation in sample.annotations:
            label = annotation.get("label")
            if label and label not in class_names:
                class_names.append(label)

    class_names.sort()
    class_index = {name: idx for idx, name in enumerate(class_names)}

    for sample in samples:
        dest_image = images_dir / sample.image_path.name
        shutil.copy2(sample.image_path, dest_image)
        try:
            from PIL import Image
            with Image.open(sample.image_path) as img:
                width, height = img.size
        except Exception:
            raise RuntimeError(f"Unable to read image size for {sample.image_path}")

        label_file = labels_dir / f"{sample.image_path.stem}.txt"
        label_lines: list[str] = []
        for annotation in sample.annotations:
            bbox = annotation.get("bbox")
            label = annotation.get("label")
            if not bbox or not label:
                continue
            xc, yc, w, h = normalize_bbox(bbox, width, height)
            label_lines.append(f"{class_index[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_file.write_text("\n".join(label_lines), encoding="utf-8")

    return dataset_dir, class_names


def train_yolo_seg(dataset_dir: Path, class_names: list[str], output_dir: Path, epochs: int = 20) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    result = model.train(
        data={
            "train": str(dataset_dir / "images"),
            "val": str(dataset_dir / "images"),
            "nc": len(class_names),
            "names": class_names,
        },
        epochs=epochs,
        batch=2,
        imgsz=640,
        project=str(output_dir),
        name="yolo_seg_few_shot",
    )
    return {
        "model_file": str(output_dir / "yolo_seg_few_shot" / "weights" / "best.pt"),
        "history": str(result),
    }


def build_calibration(samples: list[AnnotationSample], model_name: str, output_dir: Path) -> Path:
    vocabulary = sorted({annotation["label"] for sample in samples for annotation in sample.annotations if annotation.get("label")})
    bboxes = [annotation["bbox"] for sample in samples for annotation in sample.annotations if annotation.get("bbox")]
    avg_bbox = {
        "x0": int(sum(b["x0"] for b in bboxes) / len(bboxes)),
        "y0": int(sum(b["y0"] for b in bboxes) / len(bboxes)),
        "x1": int(sum(b["x1"] for b in bboxes) / len(bboxes)),
        "y1": int(sum(b["y1"] for b in bboxes) / len(bboxes)),
    }

    calibration = {
        "model_name": model_name,
        "vocabulary": vocabulary,
        "average_bbox": avg_bbox,
        "label_counts": dict(Counter(annotation["label"] for sample in samples for annotation in sample.annotations if annotation.get("label"))),
    }

    calibration_path = output_dir / f"{model_name}_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return calibration_path


def train_model(sample_data: list[AnnotationSample], model_name: str, output_dir: Path) -> dict[str, Any]:
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"model_name": model_name, "status": "trained", "output_dir": str(model_output_dir)}

    if model_name == "yolo_seg":
        dataset_dir, class_names = build_yolo_dataset(sample_data, model_output_dir)
        class_names_path = model_output_dir / "yolo_dataset" / "class_names.json"
        class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
        try:
            yolo_result = train_yolo_seg(dataset_dir, class_names, model_output_dir, epochs=20)
            summary["trained_model"] = yolo_result
            summary["class_names"] = class_names
        except Exception as exc:
            summary["status"] = "failed"
            summary["error"] = str(exc)
    else:
        calibration_path = build_calibration(sample_data, model_name, model_output_dir)
        summary["calibration_file"] = str(calibration_path)
        summary["calibration_prompt"] = " . ".join(sorted({annotation["label"] for sample in sample_data for annotation in sample.annotations if annotation.get("label")}))

    return summary


def evaluate_yolo_model(sample_data: list[AnnotationSample], weights_path: Path, class_names: list[str], output_dir: Path) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    model_output_dir = output_dir / "yolo_seg"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for sample in sample_data:
        predictions = model.predict(source=str(sample.image_path), save=False)
        if not predictions:
            continue

        prediction = predictions[0]
        boxes = []
        if hasattr(prediction, "boxes") and prediction.boxes is not None:
            xyxy = prediction.boxes.xyxy.cpu().numpy()
            cls = prediction.boxes.cls.cpu().numpy()
            conf = prediction.boxes.conf.cpu().numpy()
            for idx, box in enumerate(xyxy):
                x0, y0, x1, y1 = box.tolist()
                class_name = class_names[int(cls[idx])] if int(cls[idx]) < len(class_names) else str(int(cls[idx]))
                boxes.append({
                    "label": class_name,
                    "score": float(conf[idx]),
                    "bbox": {
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x1),
                        "y1": int(y1),
                    },
                })

        results.append({
            "image_path": str(sample.image_path),
            "model_name": "yolo_seg",
            "predicted_labels": boxes,
            "segmented_image_path": str(model_output_dir / sample.image_path.name),
            "triples_path": "",
            "metrics": {},
        })

    return results


def evaluate_fine_tuned_models(sample_data: list[AnnotationSample], model_names: list[str], output_dir: Path, prompt: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for model_name in model_names:
        model_dir = output_dir / model_name
        if model_name == "yolo_seg":
            weights_path = model_dir / "yolo_seg_few_shot" / "weights" / "best.pt"
            if weights_path.exists():
                class_names_path = model_dir / "yolo_dataset" / "class_names.json"
                class_names = []
                if class_names_path.exists():
                    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
                results.extend(evaluate_yolo_model(sample_data, weights_path, class_names, model_dir))
                continue

        model_output_dir = model_dir / "evaluation"
        segment_output_dir = model_output_dir / "output"
        segmenter = DinoSamImageSegmenter(prompt=prompt, output_dir=segment_output_dir, model_name=model_name)
        for sample in sample_data:
            result = segmenter.segment_and_recognize(sample.image_path)
            results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Few-shot fine-tune image segmentation and detection backends using annotated damaged engine images"
    )
    parser.add_argument("data_dir", help="Directory containing annotation JSON files and the associated training images")
    parser.add_argument(
        "--output-dir",
        default="output/fine_tuned",
        help="Directory where fine-tuning artifacts and evaluation outputs will be written",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="One or more model backends to fine-tune or calibrate",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Vocabulary prompt used for calibration and evaluation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to use for YOLO-Seg training",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_data = load_annotation_samples(data_dir)

    summary: list[dict[str, Any]] = []
    for model_name in args.models:
        print(f"Fine-tuning {model_name}...")
        model_summary = train_model(sample_data, model_name, output_dir)
        summary.append(model_summary)
        print(json.dumps(model_summary, indent=2))

    evaluation_results = evaluate_fine_tuned_models(sample_data, args.models, output_dir, args.prompt)
    evaluation_path = output_dir / "fine_tuned_evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation_results, indent=2), encoding="utf-8")

    summary_path = output_dir / "fine_tune_summary.json"
    summary_data = {
        "model_summary": summary,
        "evaluation_path": str(evaluation_path),
        "sample_count": len(sample_data),
        "models": args.models,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Fine-tuning completed. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
