#!/usr/bin/env python3
"""使用 candy.v2 資料集訓練 YOLO11 物件偵測模型。"""

from pathlib import Path

import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "candy.v2i.yolov11" / "data.yaml"
MODELS_DIR = ROOT / "models"
RUN_NAME = "candy_v2"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATA),
        epochs=80,
        imgsz=640,
        batch=8,
        patience=20,
        project=str(ROOT / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=True,
        device="cpu",
        workers=0,
        cos_lr=True,
        close_mosaic=10,
    )

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = MODELS_DIR / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n最佳模型已複製至：{best_dst}")
    else:
        print(f"\n警告：找不到 {best_src}，請手動從 runs 目錄複製 best.pt")


if __name__ == "__main__":
    main()
