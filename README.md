# Image Dataset Resizer & Compressor

## Overview

This Python utility is designed to **prepare image datasets** for evaluating **aesthetic score predictors**. It resizes and compresses images from various source folders into a unified structure with multiple resolution and quality configurations.

This allows you to test the consistency and robustness of aesthetic prediction models across a variety of image resolutions and compression levels.

---

## Example Use Case

You may want to evaluate an aesthetic model’s performance on images generated from different sources such as:

- `Stable Diffusion 3.5`
- `Flux.1 Schnell`
- `DALL·E 3`
- Real-world photographs

By placing each image source in its own folder, this tool will resize and compress each image across a variety of quality levels and save them accordingly.

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
└── README.md
```

