from pathlib import Path
import sys
import tempfile
import zipfile


MEDIA_ORDER = [
    "primary_top.png",
    "primary_single.png",
    "grounding_output.png",
    "no_target.png",
    "primary_top.png",
    "primary_top.png",
    "primary_single.png",
    "primary_single.png",
]


def media_index(name: str) -> int | None:
    if not name.startswith("ppt/media/image") or not name.endswith(".png"):
        return None
    stem = Path(name).stem
    suffix = stem.removeprefix("image")
    return 0 if suffix == "" else int(suffix) - 1


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: patch_pptx_media.py <pptx_path> <asset_dir>")

    pptx_path = Path(sys.argv[1])
    asset_dir = Path(sys.argv[2])
    media_payloads = [(asset_dir / name).read_bytes() for name in MEDIA_ORDER]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
        tmp_path = Path(tmp.name)

    with zipfile.ZipFile(pptx_path, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            idx = media_index(info.filename)
            if idx is not None and idx < len(media_payloads):
                zout.writestr(info, media_payloads[idx])
            else:
                zout.writestr(info, zin.read(info.filename))

    tmp_path.replace(pptx_path)
    print(f"Patched PPTX media in {pptx_path}")


if __name__ == "__main__":
    main()
