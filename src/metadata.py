import json
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def make_onehot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

def resolve_existing_columns(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Tuple[List[str], List[str]]:
    existing_num_cols = [c for c in num_cols if c in df.columns]
    existing_cat_cols = [c for c in cat_cols if c in df.columns]

    return existing_num_cols, existing_cat_cols

def build_metadata_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor

def fit_transform_metadata(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
):
    preprocessor = build_metadata_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    features = preprocessor.fit_transform(df)

    if hasattr(features, "toarray"):
        features = features.toarray()

    features = features.astype(np.float32)

    return preprocessor, features

def transform_metadata(
    preprocessor,
    df: pd.DataFrame,
):
    features = preprocessor.transform(df)

    if hasattr(features, "toarray"):
        features = features.toarray()

    features = features.astype(np.float32)

    return features

def get_metadata_feature_names(
    preprocessor,
    num_cols: List[str],
    cat_cols: List[str],
) -> List[str]:
    feature_names = []

    feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        onehot = cat_pipeline.named_steps["onehot"]

        cat_feature_names = onehot.get_feature_names_out(cat_cols)
        feature_names.extend(cat_feature_names.tolist())

    return feature_names

def build_processed_dataframe(
    original_df: pd.DataFrame,
    features: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    base_cols = [
        "isic_id",
        "patient_id",
    ]

    if "target" in original_df.columns:
        base_cols.append("target")

    base_df = original_df[base_cols].reset_index(drop=True)

    feature_df = pd.DataFrame(
        features,
        columns=feature_names,
    )

    processed_df = pd.concat(
        [base_df, feature_df],
        axis=1,
    )

    return processed_df

def save_preprocessor(preprocessor, path: str):
    joblib.dump(preprocessor, path)

def load_preprocessor(path: str):
    return joblib.load(path)

def save_metadata_info(
    path: str,
    num_cols: List[str],
    cat_cols: List[str],
    feature_names: List[str],
):
    info = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": feature_names,
        "meta_dim": len(feature_names),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def load_metadata_info(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)