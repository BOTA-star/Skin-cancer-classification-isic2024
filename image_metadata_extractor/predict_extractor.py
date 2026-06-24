import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import ImageMetadataRegressor
from transforms import get_eval_transform
from utils import inverse_scale, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="predicted_metadata.json")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    target_cols = ckpt["target_cols"]
    target_stats = ckpt["target_stats"]
    image_size = ckpt.get("image_size", 224)
    backbone = ckpt.get("backbone", "mobilenet_v3_small")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageMetadataRegressor(
        num_outputs=len(target_cols),
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    transform = get_eval_transform(image_size)
    x = transform(image).unsqueeze(0).to(device)

    pred_scaled = model(x).cpu().numpy()
    pred = inverse_scale(pred_scaled, target_cols, target_stats)[0]

    result = {col: float(pred[i]) for i, col in enumerate(target_cols)}
    save_json(result, args.output_json)

    print(f"Saved: {args.output_json}")
    print(result)


if __name__ == "__main__":
    main()
