from pathlib import Path
import sys

from PIL import Image, ImageOps


def crop_with_border(image, box, out_path, border=0):
    crop = image.crop(box)
    if border:
        crop = ImageOps.expand(crop, border=border, fill="white")
    crop.save(out_path)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: extract_flowchart_assets.py <source_png> <output_dir>")

    source = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(source).convert("RGB")

    crops = {
        "primary_top.png": (70, 295, 245, 475),
        "primary_single.png": (70, 900, 220, 1070),
        "grounding_output.png": (2465, 330, 2800, 575),
        "no_target.png": (2565, 1015, 2800, 1240),
    }

    for name, box in crops.items():
        crop_with_border(image, box, out_dir / name, border=2)

    print(f"Extracted {len(crops)} assets to {out_dir}")


if __name__ == "__main__":
    main()
