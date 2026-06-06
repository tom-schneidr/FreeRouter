from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

from app.ui.brand import FAVICON_PATH, PROJECT_ROOT

ICON_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def icon_targets(project_root: Path | None = None) -> tuple[Path, Path]:
    root = project_root or PROJECT_ROOT
    return (
        root / "apps" / "desktop" / "src-tauri" / "icons" / "icon.ico",
        root / "data" / "freerouter.ico",
    )


def npx_executable() -> str:
    for candidate in ("npx.cmd", "npx.exe", "npx"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError("Could not find npx. Install Node.js and run npm install first.")


def build_icon_image(size: int = 256) -> Image.Image:
    image = Image.open(FAVICON_PATH).convert("RGBA")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2), image)
    return canvas


def write_icon(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [build_icon_image(size) for size, _ in ICON_SIZES]
    images[0].save(
        path,
        format="ICO",
        sizes=[size for size in ICON_SIZES],
        append_images=images[1:],
    )
    return path


def sync_tauri_icons(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    desktop_dir = root / "apps" / "desktop"
    icons_dir = desktop_dir / "src-tauri" / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="freerouter-icons-") as temporary_directory:
        generated_dir = Path(temporary_directory)
        subprocess.run(
            [
                npx_executable(),
                "tauri",
                "icon",
                str(FAVICON_PATH.resolve()),
                "-o",
                str(generated_dir),
            ],
            cwd=desktop_dir,
            check=True,
        )
        _sync_generated_icons(generated_dir, icons_dir)

    tauri_icon = icons_dir / "icon.ico"
    if not tauri_icon.exists():
        raise FileNotFoundError(f"Tauri icon generation did not produce {tauri_icon}")
    return tauri_icon


def _sync_generated_icons(generated_dir: Path, icons_dir: Path) -> None:
    for generated in generated_dir.rglob("*"):
        if not generated.is_file():
            continue
        target = icons_dir / generated.relative_to(generated_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _icons_equivalent(generated, target):
            continue
        shutil.copy2(generated, target)


def _icons_equivalent(left: Path, right: Path) -> bool:
    if filecmp.cmp(left, right, shallow=False):
        return True
    try:
        with Image.open(left) as left_image, Image.open(right) as right_image:
            if left_image.size != right_image.size:
                return False
            difference = ImageChops.difference(
                left_image.convert("RGBA"),
                right_image.convert("RGBA"),
            )
            return all(channel_max == 0 for _, channel_max in difference.getextrema())
    except (OSError, ValueError):
        return False


def sync_icon_targets(project_root: Path | None = None) -> list[Path]:
    root = project_root or PROJECT_ROOT
    tauri_icon, shortcut_icon = icon_targets(root)
    synced_icon = sync_tauri_icons(root)
    if synced_icon.resolve() != tauri_icon.resolve():
        shutil.copy2(synced_icon, tauri_icon)
    shortcut_icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tauri_icon, shortcut_icon)
    return [tauri_icon, shortcut_icon]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the FreeRouter desktop icon.")
    parser.add_argument("--output", type=Path, action="append")
    parser.add_argument("--sync-all", action="store_true")
    args = parser.parse_args()

    if args.sync_all:
        for target in sync_icon_targets():
            print(target)
        return

    targets = args.output or [Path("data/freerouter.ico")]
    for target in targets:
        print(write_icon(target))


if __name__ == "__main__":
    main()
