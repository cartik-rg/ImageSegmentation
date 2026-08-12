# KnowledgeGraphImageSegmentation

This sibling project contains the image segmentation and object recognition pipelines extracted from the original KnowledgeGraphCreator repository.

## Project structure

- `src/image_processing/image_segmenter.py`: shared image segmentation and object-recognition service for multiple vision backends
- `scripts/run_image_segmentation_with_sam_dino.py`: general-purpose image segmentation comparison entry point
- `scripts/run_all_models_comparison.py`: bulk comparison runner for all supported backends
- `scripts/few_shot_finetune_models.py`: few-shot fine-tuning script for annotated damaged engine images
- `input/image/`: image dataset used for segmentation experiments
- `input/damaged_engines/annotations/`: annotation JSON files for the few-shot training dataset
- `input/damaged_engines/images/`: image files referenced by the few-shot annotations
- `output/`: generated segmentation outputs, model artifacts, and evaluation results
- `yolov8n.pt`: local YOLOv8 weight file used for fine-tuning
- `runs/`: YOLO training and detection runtime artifacts

## Usage

Install the project in editable mode:

- python -m venv .venv
- .venv\Scripts\activate
- pip install -e .

Run image segmentation or model comparison:

- python scripts/run_all_models_comparison.py input/image --output-dir output

Run few-shot fine-tuning with annotated damaged engine images:

- python scripts/few_shot_finetune_models.py input/damaged_engines --output-dir output/fine_tuned
