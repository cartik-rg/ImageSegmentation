from __future__ import annotations

import argparse
import json
import sys
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
    "anomalib_patchcore",
]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_available_models(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        return []

    models: list[str] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        calibration_file = next(child.glob("*_calibration.json"), None)
        if calibration_file is not None:
            models.append(child.name)
    return models


def load_ontology(ontology_dir: Path, ontology_name: str) -> dict[str, Any]:
    ontology_path = ontology_dir / ontology_name
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
    return json.loads(ontology_path.read_text(encoding="utf-8"))


def find_iri_for_entity(ontology_payload: dict[str, Any], entity_id: str) -> str | None:
    entities = ontology_payload.get("entities") or ontology_payload.get("classes") or []
    for entity in entities:
        if str(entity.get("id", "")).lower() == str(entity_id).lower():
            return f"http://www.robbins-gioia.com/{ontology_payload.get('@context', {}).get('@vocab', '').replace('#', '').replace(':', '') }"  # pragma: no cover
    return None


def build_ontology_reference(ontology_payload: dict[str, Any], entity_id: str) -> dict[str, str]:
    entity_id = entity_id.strip()
    ontology_prefix = "http://www.robbins-gioia.com/aircraft-ontology#"
    if "damage" in str(ontology_payload).lower() or "classes" in ontology_payload:
        ontology_prefix = "http://www.robbins-gioia.com/aircraft-damage-ontology#"

    return {
        "id": entity_id,
        "iri": f"{ontology_prefix}{entity_id}",
        "source": "aircraft-ontology" if "aircraft" in ontology_prefix else "aircraft-damage-ontology",
    }


def pick_object_and_damage(label: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    lowered = (label or "").lower()

    if "wing" in lowered:
        object_ref = {"id": "WingSystem", "iri": "http://www.robbins-gioia.com/aircraft-ontology#WingSystem", "source": "aircraft-ontology"}
        part_ref = {"id": "WingBox", "iri": "http://www.robbins-gioia.com/aircraft-ontology#WingBox", "source": "aircraft-ontology"}
    elif "propeller" in lowered or "blade" in lowered or "fan" in lowered:
        object_ref = {"id": "Engine", "iri": "http://www.robbins-gioia.com/aircraft-ontology#Engine", "source": "aircraft-ontology"}
        part_ref = {"id": "FanBlade", "iri": "http://www.robbins-gioia.com/aircraft-ontology#FanBlade", "source": "aircraft-ontology"}
    elif "wire" in lowered or "electrical" in lowered:
        object_ref = {"id": "ElectricalWire", "iri": "http://www.robbins-gioia.com/aircraft-ontology#ElectricalWire", "source": "aircraft-ontology"}
        part_ref = {"id": "WireHarness", "iri": "http://www.robbins-gioia.com/aircraft-ontology#WireHarness", "source": "aircraft-ontology"}
    elif "landing" in lowered or "gear" in lowered:
        object_ref = {"id": "LandingGearSystem", "iri": "http://www.robbins-gioia.com/aircraft-ontology#LandingGearSystem", "source": "aircraft-ontology"}
        part_ref = {"id": "MainLandingGear", "iri": "http://www.robbins-gioia.com/aircraft-ontology#MainLandingGear", "source": "aircraft-ontology"}
    else:
        object_ref = {"id": "Engine", "iri": "http://www.robbins-gioia.com/aircraft-ontology#Engine", "source": "aircraft-ontology"}
        part_ref = {"id": "FanBlade", "iri": "http://www.robbins-gioia.com/aircraft-ontology#FanBlade", "source": "aircraft-ontology"}

    if "corrosion" in lowered:
        damage_ref = {"id": "CorrosionDamage", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#CorrosionDamage", "source": "aircraft-damage-ontology"}
        damage_subtype_ref = {"id": "PittingCorrosion", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#PittingCorrosion", "source": "aircraft-damage-ontology"}
    elif "crack" in lowered:
        damage_ref = {"id": "SurfaceDamage", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#SurfaceDamage", "source": "aircraft-damage-ontology"}
        damage_subtype_ref = {"id": "Crack", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#Crack", "source": "aircraft-damage-ontology"}
    elif "dent" in lowered or "deform" in lowered or "bend" in lowered or "warping" in lowered:
        damage_ref = {"id": "DeformationDamage", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#DeformationDamage", "source": "aircraft-damage-ontology"}
        damage_subtype_ref = {"id": "Dent", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#Dent", "source": "aircraft-damage-ontology"}
    elif "impact" in lowered or "broken" in lowered or "foreign" in lowered:
        damage_ref = {"id": "ImpactDamage", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#ImpactDamage", "source": "aircraft-damage-ontology"}
        damage_subtype_ref = {"id": "ForeignObjectDamage", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#ForeignObjectDamage", "source": "aircraft-damage-ontology"}
    elif "gouge" in lowered or "scratch" in lowered or "score" in lowered:
        damage_ref = {"id": "SurfaceDamage", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#SurfaceDamage", "source": "aircraft-damage-ontology"}
        damage_subtype_ref = {"id": "Gouge", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#Gouge", "source": "aircraft-damage-ontology"}
    else:
        damage_ref = {"id": "SurfaceDamage", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#SurfaceDamage", "source": "aircraft-damage-ontology"}
        damage_subtype_ref = {"id": "ScoreMark", "iri": "http://www.robbins-gioia.com/aircraft-damage-ontology#ScoreMark", "source": "aircraft-damage-ontology"}

    return object_ref, part_ref, damage_ref, damage_subtype_ref


def bbox_to_dict(raw_bbox: Any) -> dict[str, int]:
    if isinstance(raw_bbox, dict):
        x0 = int(raw_bbox.get("x0", 0))
        y0 = int(raw_bbox.get("y0", 0))
        x1 = int(raw_bbox.get("x1", 0))
        y1 = int(raw_bbox.get("y1", 0))
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        x0, y0, x1, y1 = [int(v) for v in raw_bbox]
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}


def make_annotation_payload(image_path: Path, labels: list[dict[str, Any]]) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []
    for entry in labels:
        label = str(entry.get("label") or "unknown object")
        object_ref, part_ref, damage_ref, damage_subtype_ref = pick_object_and_damage(label)
        bbox = bbox_to_dict(entry.get("bbox"))
        score = float(entry.get("score", 0.0) or 0.0)
        score = max(0.0, min(1.0, score))

        annotations.append(
            {
                "label": label,
                "object": object_ref,
                "part_of": part_ref,
                "damage": damage_ref,
                "damage_subtype": damage_subtype_ref,
                "bbox": bbox,
                "confidence": round(score, 2),
            }
        )

    return {
        "image_id": image_path.stem,
        "file_name": image_path.name,
        "source": {
            "aircraft_ontology": "ontology/aircraft-ontology.json",
            "damage_ontology": "ontology/aircraft-damage-ontology.json",
        },
        "annotations": annotations,
    }


def infer_label_set_for_model(model_dir: Path, model_name: str) -> str:
    calibration_path = next(model_dir.glob(f"{model_name}_calibration.json"), None)
    if calibration_path is None:
        calibration_path = next(model_dir.glob("*_calibration.json"), None)
    if calibration_path is not None:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        prompt = payload.get("calibration_prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        vocab = payload.get("vocabulary")
        if isinstance(vocab, list) and vocab:
            return " . ".join(str(item) for item in vocab)
    return (
        "damaged aircraft engine . engine damage . fan blade . crack . corrosion . "
        "broken blade . deformation . dent . aircraft propulsion system"
    )


def collect_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_files = [
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    if not image_files:
        raise FileNotFoundError(f"No supported image files were found in {input_dir}")
    return sorted(image_files)


def run_model_on_images(model_name: str, image_dir: Path, output_dir: Path, ontology_dir: Path) -> list[dict[str, Any]]:
    model_root = Path(output_dir).resolve() / "fine-tuned" / model_name
    if not model_root.exists():
        model_root = Path(output_dir).resolve() / model_name

    prompt = infer_label_set_for_model(model_root, model_name)
    temp_run_dir = Path(output_dir).resolve() / "__model_runs__" / model_name
    temp_run_dir.mkdir(parents=True, exist_ok=True)

    segmenter = DinoSamImageSegmenter(prompt=prompt, output_dir=temp_run_dir, model_name=model_name)
    segments: list[dict[str, Any]] = []

    for image_path in collect_image_files(image_dir):
        result = segmenter.segment_and_recognize(image_path)
        source_image = Path(result["segmented_image_path"])
        target_annotation_dir = Path(output_dir).resolve() / "annotations"
        target_annotation_dir.mkdir(parents=True, exist_ok=True)

        annotation_payload = make_annotation_payload(image_path, result.get("predicted_labels", []))
        annotation_path = target_annotation_dir / f"{image_path.stem}_output_annotations.json"
        annotation_path.write_text(json.dumps(annotation_payload, indent=2), encoding="utf-8")

        segments.append(
            {
                "image_name": image_path.name,
                "model_name": model_name,
                "segmented_image_path": str(source_image),
                "annotation_path": str(annotation_path),
                "confidence_count": len(annotation_payload["annotations"]),
            }
        )

    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fine-tuned segmentation models on a directory of images and export segmented images plus JSON annotations."
    )
    parser.add_argument("image_dir", help="Directory containing input images to segment")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root output directory (default: output)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of fine-tuned models to execute. Defaults to all models available under output/fine-tuned.",
    )
    parser.add_argument(
        "--ontology-dir",
        default="ontology",
        help="Directory containing aircraft and damage ontology JSON files",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ontology_dir = Path(args.ontology_dir).resolve()

    model_root = (PROJECT_ROOT / "output" / "fine-tuned").resolve()
    available_models = discover_available_models(model_root)
    selected_models = args.models or available_models or DEFAULT_MODELS
    selected_models = [model for model in selected_models if model in available_models or model in DEFAULT_MODELS]

    if not selected_models:
        raise FileNotFoundError("No fine-tuned model directories were found in output/fine-tuned")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for model_name in selected_models:
        model_dir = model_root / model_name
        if not model_dir.exists():
            continue
        try:
            model_summary = run_model_on_images(model_name, image_dir, output_dir, ontology_dir)
            summary.extend(model_summary)
            print(f"Completed model: {model_name}")
        except Exception as exc:  # pragma: no cover - keeps CLI resilient when one model fails
            print(f"Model {model_name} failed: {exc}")

    summary_path = output_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Inference summary written to {summary_path}")


if __name__ == "__main__":
    main()
