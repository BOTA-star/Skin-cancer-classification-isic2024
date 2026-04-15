# ISIC 2024 Skin Cancer Classification

## How to Run
1. Download dataset from:
https://challenge2024.isic-archive.com/

2. Place the dataset into the correct directory:
data/raw_data/

3. Run notebooks step-by-step:
- 00_extract_data.ipynb
- 01_data_integrity.ipynb
- 02_train_val_split.ipynb
- ...
- 08_test_training.ipynb

4. For quick testing:
- Set `flag = True` in `08_test_training.ipynb`

5. For full training:
- Set `flag = False`

## .py vs .ipynb
Jupyter notebooks (.ipynb) are used for experimentation and step-by-step execution
Python scripts (.py) are used for reusable modules and production-ready code

## Data Note
Dataset paths may vary depending on your local storage structure.
Please update the data directory paths in the notebooks or configuration files according to your environment.
Example:
* Change `image_dir` to match your local dataset location
* Ensure `train.csv` and `val.csv` point to the correct paths

## Debug Mode
notebooks/08_test_training.ipynb:
* Set flag = True for quick testing with limited batches.
* Set flag = False for full training.

## 14/4/2026 Summary
Model - Forward Pass - Training Loop - Debug Mode
* Implemented ResNet-based model for image classification
* Verified model forward pass with DataLoader
* Built training and validation pipeline
* Applied weighted loss to handle class imbalance
* Added debug mode for fast experimentation
Next: Train on full dataset and add evaluation metrics (AUC)
