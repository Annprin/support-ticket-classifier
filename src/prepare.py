# src/prepare.py
import argparse
from pathlib import Path
import joblib
import logging

from src.config import load_config
from src.data_loader import load_data, preprocess_data, create_preprocessor
from src.utils import save_processed_np

logger = logging.getLogger(__name__)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    raw_path = cfg["data"].get("raw_path", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv") 
    out_dir = Path(cfg["data"].get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(raw_path)
    preprocessor = create_preprocessor(cfg)
    train_df, val_df, y_train, y_val = preprocess_data(df, cfg)

    save_processed_np(out_dir, train_df, val_df, y_train, y_val)
    logger.info(f"Данные сохранены в {out_dir}")

if __name__ == "__main__":
    main()