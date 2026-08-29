#!/usr/bin/env python3
"""Visualize BONAI building boxes and roof masks from the generated COCO data."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SPLITS = (
    (
        "TRAIN",
        "instances_train.json",
        Path("trainval/images"),
    ),
    (
        "VAL",
        "instances_val.json",
        Path("test"),
    ),
)
TARGET_INSTANCE_COUNTS = (8, 18, 30)
BACKGROUND = (24, 27, 33)
HEADER = (34, 38, 46)
TEXT = (240, 243, 248)
SUBTEXT = (188, 195, 207)
BUILDING_BOX = (255, 196, 0, 255)
ROOF_FILL = (0, 220, 255, 76)
ROOF_OUTLINE = (0, 235, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/lzq/dataset/BONAI"),
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help=(
            "Generated COCO directory "
            "(default: ROOT/coco_building_bbox_roof_mask)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/bonai_building_bbox_roof_mask_preserve_bbox_R50_bs2_50ep_512/"
            "annotation_check_train_val.png"
        ),
    )
    parser.add_argument("--tile-size", type=int, default=480)
    return parser.parse_args()


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def polygon_bounds(segmentation):
    xs = [
        coordinate
        for polygon in segmentation
        for coordinate in polygon[0::2]
    ]
    ys = [
        coordinate
        for polygon in segmentation
        for coordinate in polygon[1::2]
    ]
    return min(xs), min(ys), max(xs), max(ys)


def box_roof_margin(annotation):
    x, y, width, height = annotation["bbox"]
    roof_x0, roof_y0, roof_x1, roof_y1 = polygon_bounds(
        annotation["segmentation"]
    )
    margins = (
        roof_x0 - x,
        roof_y0 - y,
        x + width - roof_x1,
        y + height - roof_y1,
    )
    return sum(max(0.0, margin) for margin in margins)


def select_images(images, annotations_by_image):
    candidates = []
    for image in images:
        annotations = [
            annotation
            for annotation in annotations_by_image.get(image["id"], ())
            if not annotation["iscrowd"]
        ]
        if not annotations:
            continue
        average_margin = sum(
            box_roof_margin(annotation) for annotation in annotations
        ) / len(annotations)
        candidates.append((image, annotations, average_margin))

    selected = []
    used_image_ids = set()
    for target_count in TARGET_INSTANCE_COUNTS:
        available = [
            candidate
            for candidate in candidates
            if candidate[0]["id"] not in used_image_ids
        ]
        chosen = min(
            available,
            key=lambda candidate: (
                abs(len(candidate[1]) - target_count),
                -candidate[2],
                candidate[0]["id"],
            ),
        )
        selected.append(chosen)
        used_image_ids.add(chosen[0]["id"])
    return selected


def scaled_points(polygon, scale_x, scale_y):
    return [
        (
            polygon[index] * scale_x,
            polygon[index + 1] * scale_y,
        )
        for index in range(0, len(polygon), 2)
    ]


def render_tile(
    image_path,
    image_info,
    annotations,
    split_name,
    tile_size,
    title_font,
    detail_font,
):
    source = Image.open(image_path).convert("RGB")
    source_width, source_height = source.size
    image = source.resize(
        (tile_size, tile_size),
        resample=Image.Resampling.BILINEAR,
    ).convert("RGBA")
    scale_x = tile_size / source_width
    scale_y = tile_size / source_height

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for annotation in annotations:
        for polygon in annotation["segmentation"]:
            points = scaled_points(polygon, scale_x, scale_y)
            overlay_draw.polygon(points, fill=ROOF_FILL)
            overlay_draw.line(
                points + [points[0]],
                fill=ROOF_OUTLINE,
                width=2,
                joint="curve",
            )
    image = Image.alpha_composite(image, overlay)

    image_draw = ImageDraw.Draw(image)
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        image_draw.rectangle(
            (
                x * scale_x,
                y * scale_y,
                (x + width) * scale_x,
                (y + height) * scale_y,
            ),
            outline=BUILDING_BOX,
            width=3,
        )

    title_height = 42
    tile = Image.new(
        "RGB",
        (tile_size, tile_size + title_height),
        HEADER,
    )
    tile.paste(image.convert("RGB"), (0, title_height))
    title_draw = ImageDraw.Draw(tile)
    title_draw.text(
        (10, 5),
        split_name,
        font=title_font,
        fill=TEXT,
    )
    file_name = image_info["file_name"]
    if len(file_name) > 35:
        file_name = "{}...{}".format(file_name[:17], file_name[-15:])
    title_draw.text(
        (82, 6),
        file_name,
        font=detail_font,
        fill=SUBTEXT,
    )
    title_draw.rectangle(
        (tile_size - 100, 0, tile_size, title_height),
        fill=HEADER,
    )
    title_draw.text(
        (tile_size - 92, 6),
        "{} objs".format(len(annotations)),
        font=detail_font,
        fill=TEXT,
    )
    return tile


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    annotation_dir = (
        args.annotation_dir.expanduser().resolve()
        if args.annotation_dir is not None
        else root / "coco_building_bbox_roof_mask"
    )
    output = args.output.expanduser().resolve()

    title_font = load_font(19, bold=True)
    detail_font = load_font(14)
    legend_font = load_font(21, bold=True)
    tile_size = args.tile_size
    tile_title_height = 42
    legend_height = 58
    gap = 8

    selected_by_split = []
    for split_name, json_name, image_relative in SPLITS:
        dataset = json.loads(
            (annotation_dir / json_name).read_text(encoding="utf-8")
        )
        annotations_by_image = defaultdict(list)
        for annotation in dataset["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)
        selected = select_images(
            dataset["images"],
            annotations_by_image,
        )
        selected_by_split.append(
            (split_name, root / image_relative, selected)
        )

    columns = len(TARGET_INSTANCE_COUNTS)
    rows = len(SPLITS)
    canvas_width = columns * tile_size + (columns - 1) * gap
    canvas_height = (
        legend_height
        + rows * (tile_size + tile_title_height)
        + (rows - 1) * gap
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)

    legend_draw = ImageDraw.Draw(canvas)
    legend_draw.rectangle((0, 0, canvas_width, legend_height), fill=HEADER)
    legend_draw.rectangle(
        (20, 18, 62, 38),
        outline=BUILDING_BOX,
        width=3,
    )
    legend_draw.text(
        (74, 14),
        "building_bbox",
        font=legend_font,
        fill=TEXT,
    )
    legend_draw.rectangle(
        (270, 18, 312, 38),
        fill=ROOF_FILL[:3],
        outline=ROOF_OUTLINE,
        width=2,
    )
    legend_draw.text(
        (324, 14),
        "roof_mask",
        font=legend_font,
        fill=TEXT,
    )
    legend_draw.text(
        (canvas_width - 420, 16),
        "BONAI corrected annotation check",
        font=detail_font,
        fill=SUBTEXT,
    )

    for row, (split_name, image_root, selected) in enumerate(
        selected_by_split
    ):
        y = legend_height + row * (
            tile_size + tile_title_height + gap
        )
        for column, (image_info, annotations, _) in enumerate(selected):
            image_path = image_root / image_info["file_name"]
            tile = render_tile(
                image_path,
                image_info,
                annotations,
                split_name,
                tile_size,
                title_font,
                detail_font,
            )
            x = column * (tile_size + gap)
            canvas.paste(tile, (x, y))
            print(
                "{}: {} ({} usable instances)".format(
                    split_name,
                    image_info["file_name"],
                    len(annotations),
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
