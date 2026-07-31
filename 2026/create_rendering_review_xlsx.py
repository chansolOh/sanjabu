from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


ROOT_DIR = Path("/nas/ochansol/3d_model/peel3_scan_data_2026")
EXCEL_CELL_IMAGE_XLSX = ROOT_DIR / "peel3_scan_rendering_review_excel_cell_images.xlsx"
GOOGLE_COMPATIBLE_XLSX = ROOT_DIR / "peel3_scan_rendering_review_google_compatible.xlsx"

VIEW_FILES = {
    "Front": "front.png",
    "Side": "side.png",
    "Rear": "rear.png",
    "Top": "top.png",
    "Front-quarter": "front_quarter.png",
    "Rear-quarter": "rear_quarter.png",
}

ROW_HEIGHT_PX = 150
IMAGE_COL_WIDTH_PX = 190
NUMBER_COL_WIDTH_PX = 70
NAME_COL_WIDTH_PX = 220
USABLE_COL_WIDTH_PX = 170
FEEDBACK_COL_WIDTH_PX = 260
THUMBNAIL_MAX_SIZE = (IMAGE_COL_WIDTH_PX - 10, ROW_HEIGHT_PX - 10)
JPEG_QUALITY = 88
OVERWRITE = True


def require_xlsxwriter():
    try:
        import xlsxwriter
    except ImportError as exc:
        raise SystemExit(
            "xlsxwriter is required. Install it with:\n"
            "  uv pip install xlsxwriter\n"
            "or:\n"
            "  python3 -m pip install xlsxwriter"
        ) from exc
    return xlsxwriter


def pixels_to_column_width(pixels):
    if pixels <= 12:
        return pixels / 12.0
    return (pixels - 5) / 7.0


def pixels_to_row_height(pixels):
    return pixels * 0.75


def find_object_folders(root_dir):
    return sorted(path for path in root_dir.iterdir() if path.is_dir())


def has_any_rendering(object_dir):
    rendering_dir = object_dir / "rendering"
    return any((rendering_dir / filename).exists() for filename in VIEW_FILES.values())


def make_thumbnail(source_path, output_path):
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image.thumbnail(THUMBNAIL_MAX_SIZE, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", THUMBNAIL_MAX_SIZE, "white")
        left = (THUMBNAIL_MAX_SIZE[0] - image.width) // 2
        top = (THUMBNAIL_MAX_SIZE[1] - image.height) // 2
        canvas.paste(image, (left, top))
        canvas.save(output_path, "JPEG", quality=JPEG_QUALITY, optimize=True)


def insert_cell_image(worksheet, row, col, image_path, use_excel_cell_images):
    if use_excel_cell_images and hasattr(worksheet, "embed_image"):
        worksheet.embed_image(row, col, str(image_path))
        return

    worksheet.insert_image(
        row,
        col,
        str(image_path),
        {
            "x_offset": 5,
            "y_offset": 5,
            "object_position": 1,
        },
    )


def write_review_workbook(xlsxwriter, output_xlsx, object_folders, use_excel_cell_images):
    if output_xlsx.exists() and not OVERWRITE:
        raise FileExistsError(f"output exists: {output_xlsx}")

    workbook = xlsxwriter.Workbook(str(output_xlsx))
    worksheet = workbook.add_worksheet("rendering_review")

    header_fmt = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#FFF2CC",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )
    number_fmt = workbook.add_format(
        {
            "bold": True,
            "font_size": 18,
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        }
    )
    text_fmt = workbook.add_format({"valign": "vcenter", "border": 1})
    image_cell_fmt = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    usable_fmt = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    usable_o_fmt = workbook.add_format(
        {
            "bg_color": "#CFE2F3",
            "font_color": "#1F4E79",
            "bold": True,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )
    usable_x_fmt = workbook.add_format(
        {
            "bg_color": "#F4CCCC",
            "font_color": "#990000",
            "bold": True,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )
    feedback_fmt = workbook.add_format({"border": 1, "valign": "top", "text_wrap": True})
    missing_fmt = workbook.add_format(
        {
            "font_color": "#9E9E9E",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        }
    )

    columns = ["번호", "물체명", *VIEW_FILES.keys(), "사용 가능 여부", "피드백"]
    first_image_col = 2
    last_image_col = first_image_col + len(VIEW_FILES) - 1
    usable_col = last_image_col + 1
    feedback_col = usable_col + 1

    for col, title in enumerate(columns):
        worksheet.write(0, col, title, header_fmt)

    worksheet.set_column_pixels(0, 0, NUMBER_COL_WIDTH_PX)
    worksheet.set_column_pixels(1, 1, NAME_COL_WIDTH_PX)
    worksheet.set_column_pixels(first_image_col, last_image_col, IMAGE_COL_WIDTH_PX)
    worksheet.set_column_pixels(usable_col, usable_col, USABLE_COL_WIDTH_PX)
    worksheet.set_column_pixels(feedback_col, feedback_col, FEEDBACK_COL_WIDTH_PX)
    worksheet.set_row_pixels(0, 28)
    worksheet.freeze_panes(1, 2)
    worksheet.autofilter(0, 0, max(len(object_folders), 1), len(columns) - 1)

    worksheet.data_validation(
        1,
        usable_col,
        max(len(object_folders), 1),
        usable_col,
        {
            "validate": "list",
            "source": ["o", "x"],
            "input_message": "o or x",
        },
    )
    worksheet.conditional_format(
        1,
        usable_col,
        max(len(object_folders), 1),
        usable_col,
        {
            "type": "cell",
            "criteria": "==",
            "value": '"o"',
            "format": usable_o_fmt,
        },
    )
    worksheet.conditional_format(
        1,
        usable_col,
        max(len(object_folders), 1),
        usable_col,
        {
            "type": "cell",
            "criteria": "==",
            "value": '"x"',
            "format": usable_x_fmt,
        },
    )

    with TemporaryDirectory(prefix="rendering_review_thumbs_") as temp_dir:
        temp_dir = Path(temp_dir)
        for index, object_dir in enumerate(object_folders, start=1):
            row = index
            worksheet.set_row_pixels(row, ROW_HEIGHT_PX)
            worksheet.write_number(row, 0, index, number_fmt)
            worksheet.write(row, 1, object_dir.name, text_fmt)
            worksheet.write_blank(row, usable_col, None, usable_fmt)
            worksheet.write_blank(row, feedback_col, None, feedback_fmt)

            rendering_dir = object_dir / "rendering"
            for view_offset, (_, filename) in enumerate(VIEW_FILES.items(), start=first_image_col):
                image_path = rendering_dir / filename
                worksheet.write_blank(row, view_offset, None, image_cell_fmt)
                if not image_path.exists():
                    worksheet.write(row, view_offset, "missing", missing_fmt)
                    continue

                thumb_path = temp_dir / f"{index:04d}_{view_offset}_{image_path.stem}.jpg"
                make_thumbnail(image_path, thumb_path)
                insert_cell_image(worksheet, row, view_offset, thumb_path, use_excel_cell_images)

            if index % 25 == 0:
                print(f"processed rows: {index}/{len(object_folders)}")

        workbook.close()

    print(f"saved: {output_xlsx}")


def main():
    xlsxwriter = require_xlsxwriter()

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"root dir not found: {ROOT_DIR}")

    object_folders = [path for path in find_object_folders(ROOT_DIR) if has_any_rendering(path)]
    print(f"found rendered object folders: {len(object_folders)}")

    write_review_workbook(
        xlsxwriter,
        EXCEL_CELL_IMAGE_XLSX,
        object_folders,
        use_excel_cell_images=True,
    )
    write_review_workbook(
        xlsxwriter,
        GOOGLE_COMPATIBLE_XLSX,
        object_folders,
        use_excel_cell_images=False,
    )


if __name__ == "__main__":
    main()
