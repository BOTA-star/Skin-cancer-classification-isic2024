# Image Metadata Extractor

Module phụ dùng để train model dự đoán các `image-derived metadata` từ ảnh tổn thương da ISIC.

Mục tiêu:

```text
image -> tbp_lv_* numeric features
```

Module này được để riêng để không ảnh hưởng pipeline phân loại ung thư da chính. Sau khi train ổn có thể tích hợp lại vào Flask/web demo hoặc model multimodal chính.

## Cấu trúc dữ liệu yêu cầu

Đặt dataset giống repo chính:

```text
data/
└── raw/
    ├── train-metadata.csv
    └── ISIC_2024_Training_Input/
        ├── ISIC_0015670.jpg
        ├── ISIC_0015845.jpg
        └── ...
```

## Cài thư viện

```bash
pip install torch torchvision pandas numpy scikit-learn pillow tqdm joblib
```

## Train nhanh 50k trên Google Colab

```bash
python train_extractor.py \
  --metadata-csv /content/data/raw/train-metadata.csv \
  --image-dir /content/data/raw/ISIC_2024_Training_Input \
  --output-dir /content/outputs/image_metadata_extractor \
  --sample-size 50000 \
  --backbone mobilenet_v3_small \
  --pretrained \
  --image-size 224 \
  --epochs 5 \
  --batch-size 32
```

Nếu GPU Colab yếu, giảm batch size:

```bash
--batch-size 16
```

Nếu muốn train mạnh hơn:

```bash
--backbone efficientnet_b0
```

Còn EfficientNet-B3 nên để bạn chạy Kaggle full.

## Target columns

Mặc định script lấy toàn bộ cột numeric dạng:

```text
tbp_lv_*
```

trong metadata.

Các cột cá nhân/lâm sàng như `age_approx`, `sex`, `anatom_site_general`, `patient_id` không được train trong module này vì chúng thuộc luồng người dùng nhập hoặc quản lý hồ sơ.

## Output

Sau khi train, thư mục output sẽ có:

```text
best_extractor.pth
last_extractor.pth
target_cols.json
target_stats.json
training_log.csv
val_metrics.csv
```

## Inference thử một ảnh

```bash
python predict_extractor.py \
  --checkpoint /content/outputs/image_metadata_extractor/best_extractor.pth \
  --image-path /content/data/raw/ISIC_2024_Training_Input/ISIC_0015670.jpg \
  --output-json /content/outputs/predicted_metadata.json
```

## Lưu ý

Các feature như `tbp_lv_x`, `tbp_lv_y`, `tbp_lv_z` có thể nằm trong nhóm `tbp_lv_*`, nhưng về bản chất chúng liên quan đến hệ tọa độ/thiết bị 3D-TBP. Nếu ảnh upload ngoài thực tế chỉ là ảnh crop thông thường, các feature này có thể không được suy ra đáng tin cậy. Script vẫn hỗ trợ train nếu chúng là numeric, nhưng khi báo cáo nên nói đây là các feature phụ thuộc thiết bị và có thể cần impute trong môi trường thực tế.
