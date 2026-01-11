import argparse
from pathlib import Path
import joblib
import logging
import numpy as np

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
    train_df, val_df, y_train, y_val = preprocess_data(df, cfg)

    # save_processed_np(out_dir, train_df, val_df, y_train, y_val)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    y_train_path = out_dir / "y_train.npy"
    y_val_path = out_dir / "y_val.npy"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    np.save(y_train_path, y_train.values)
    np.save(y_val_path, y_val.values)
    logger.info(f"Данные сохранены в {out_dir}")

if __name__ == "__main__":
    main()