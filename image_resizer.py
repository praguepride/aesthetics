import os
from PIL import Image, UnidentifiedImageError

def file_resizer(
    source_folders,
    base_target_folder,
    target_size=(306, 306),
    quality_levels=(95, 85, 75, 65, 50, 25, 10)
):
    """
    Resize and compress images from multiple source folders into a structured target folder.

    Parameters:
    - source_folders (list of str): Paths to the folders containing the source images.
    - base_target_folder (str): Base path where resized images will be stored.
    - target_size (tuple of int): Size (width, height) to resize images to.
    - quality_levels (tuple of int): JPEG quality levels to apply to resized images.

    Notes:
    - Only '.jpg', '.jpeg', and '.png' images are processed.
    - Output folders are structured as: base_target_folder/jpg_{width}_{quality}q/{source_folder_name}/
    """
    image_extensions = ('.jpg', '.jpeg', '.png')

    for quality in quality_levels:
        target_folder = os.path.join(base_target_folder, f"jpg_{target_size[0]}_{quality}q")
        os.makedirs(target_folder, exist_ok=True)

        for source_folder in source_folders:
            if not os.path.isdir(source_folder):
                print(f"Warning: Source folder '{source_folder}' does not exist or is not a directory.")
                continue

            subfolder_name = os.path.basename(source_folder.rstrip("/\\"))
            output_folder = os.path.join(target_folder, subfolder_name)
            os.makedirs(output_folder, exist_ok=True)

            for filename in os.listdir(source_folder):
                if filename.lower().endswith(image_extensions):
                    input_path = os.path.join(source_folder, filename)
                    try:
                        with Image.open(input_path) as img:
                            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

                            # Ensure RGB format for saving as JPEG
                            if img_resized.mode != "RGB":
                                img_resized = img_resized.convert("RGB")

                            base_filename = os.path.splitext(filename)[0]
                            output_path = os.path.join(output_folder, f"{base_filename}.jpg")
                            img_resized.save(output_path, "JPEG", quality=quality)
                    except FileNotFoundError:
                        print(f"Error: File '{input_path}' not found.")
                    except UnidentifiedImageError:
                        print(f"Error: '{input_path}' is not a valid image file.")
                    except OSError as e:
                        print(f"OS error while processing '{input_path}': {e}")
                    except Exception as e:
                        print(f"Unexpected error processing '{input_path}': {e}")
