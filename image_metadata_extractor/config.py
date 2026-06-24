from pathlib import Path

DEFAULT_TARGET_PREFIX = "tbp_lv_"

# Các cột này không thuộc bài toán image -> metadata.
# patient_id dùng quản lý/split, age/sex/anatom_site là luồng người dùng nhập.
NEVER_TARGET_COLS = {
    "isic_id",
    "target",
    "patient_id",
    "age_approx",
    "sex",
    "anatom_site_general",
    "lesion_id",
    "iddx_full",
    "iddx_1",
    "iddx_2",
    "iddx_3",
    "iddx_4",
    "iddx_5",
    "mel_mitotic_index",
    "mel_thick_mm",
    "attribution",
    "copyright_license",
    "image_type",
    "tbp_tile_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
}

# Các feature này numeric nhưng không chắc suy ra tốt từ ảnh crop upload ngoài thực tế.
WEAK_IMAGE_TARGETS = {
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
}
