# Image Dataset Resizer & Aesthetic Scorer

## Overview

This Python utility is designed to **prepare and evaluate image datasets** for testing **aesthetic score prediction models**. It consists of two main components:

1. **Image Resizer & Compressor** – resizes and compresses images across various resolution and quality levels.
2. **Aesthetic Score Predictor** – uses CLIP and a trained MLP to generate aesthetic scores for each image.

This pipeline helps benchmark model robustness across image quality degradation and compression artifacts.

---

## Example Use Case

You may want to evaluate an aesthetic model's performance on images generated from different sources, such as:

* `Stable Diffusion 3.5`
* `Flux.1 Schnell`
* `DALL·E 3`
* Real-world photographs

By placing each image source in its own folder, this tool resizes and compresses each image into multiple variants, and then predicts aesthetic scores per configuration.

---

## Sample Folder Structure

```
project_root/
├── image_sources/
│   ├── stable_diffusion_3.5/
│   ├── flux_1_schnell/
│   ├── dalle_3/
│   └── real_images/
│
├── processed_images/
│   ├── jpg_306_95q/
│   │   ├── stable_diffusion_3.5/
│   │   ├── flux_1_schnell/
│   │   └── ...
│   ├── jpg_306_85q/
│   └── ...
│
├── resize_script.py
├── aesthetics_predictor.py
└── README.md
```

---

## Image Resizer & Compressor

### Description

Resizes images to a fixed dimension (default: `306x306`) and saves multiple JPEG-compressed versions using different quality levels.

### Usage Example

```python
from resize_script import file_resizer

source_folders = [
    "image_sources/stable_diffusion_3.5",
    "image_sources/flux_1_schnell",
    "image_sources/dalle_3",
    "image_sources/real_images"
]

file_resizer(source_folders, base_target_folder="processed_images")
```

---

## Aesthetic Score Predictor

### Description

Uses a pretrained MLP model (e.g. trained on `sac+logos+ava1`) and CLIP embeddings to compute a scalar aesthetic score per image.

### Command Line Usage

```bash
python aesthetics_predictor.py \
  --model_path path/to/aesthetic_model.pth \
  --input_root /path/to/processed_images \
  --subfolders jpg_306_10q jpg_306_25q jpg_306_50q \
  --output_prefix face_scores
```

### Arguments

* `--model_path` – Path to the `.pth` file containing trained model weights
* `--input_root` – Directory containing subfolders of processed images
* `--subfolders` – List of subfolder names to evaluate
* `--output_prefix` – Optional output CSV prefix (default: `predictions`)

---

## Requirements

Install dependencies with:

```bash
pip install torch torchvision pytorch-lightning ftfy regex tqdm pandas Pillow git+https://github.com/openai/CLIP.git
```

---

## Output

For each subfolder, a CSV will be generated containing:

* `SOURCE`: Parent folder name (e.g. model origin)
* `FILENAME`: Image file name
* `FILESIZE`: Image file size in bytes
* `SCORE`: Predicted aesthetic score

---

## License

Apache 2.0 License. See `LICENSE` file for details.
