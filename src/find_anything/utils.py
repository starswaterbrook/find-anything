import os
from pathlib import Path


# utility for getting image paths for a specific dir structure
# objects /
#         <object_name> /
#                     ref /  *.<ext>
#                     target /  *.<ext>
def get_object_image_paths(object_name: str, kind: str = "ref") -> list[str]:
    allowed = {"ref", "target"}
    if kind not in allowed:
        raise ValueError(f"kind must be one of {allowed}, got {kind}")  # noqa: TRY003, EM102

    base = Path("objects") / object_name
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Object folder not found: {base}")  # noqa: TRY003, EM102

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: list[str] = []

    def _collect(dirpath: Path) -> None:
        if not dirpath.exists() or not dirpath.is_dir():
            return
        for p in sorted(dirpath.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                rel = Path(os.path.relpath(p)).as_posix()
                if not rel.startswith("./"):
                    rel = "./" + rel
                out.append(rel)

    if kind in ("ref", "all"):
        _collect(base / "ref")
    if kind in ("target", "all"):
        _collect(base / "target")

    return out
