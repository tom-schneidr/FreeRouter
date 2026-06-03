from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def build_icon_image(size: int = 256) -> Image.Image:
    scale = size / 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def box(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(round(value * scale) for value in values)

    draw.rounded_rectangle(box((5, 5, 59, 59)), radius=round(15 * scale), fill="#2563eb")
    draw.rounded_rectangle(box((13, 16, 51, 48)), radius=round(8 * scale), fill="#07111f")
    draw.line(box((20, 24, 30, 32, 20, 40)), fill="#93c5fd", width=round(4 * scale), joint="curve")
    draw.line(box((34, 40, 46, 40)), fill="#22c55e", width=round(4 * scale))
    draw.rounded_rectangle(box((36, 19, 46, 29)), radius=round(3 * scale), fill="#38bdf8")
    return image


def write_icon(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = build_icon_image(256)
    image.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the FreeRouter desktop icon.")
    parser.add_argument("--output", type=Path, default=Path("data/freerouter.ico"))
    args = parser.parse_args()
    target = write_icon(args.output)
    print(target)


if __name__ == "__main__":
    main()
