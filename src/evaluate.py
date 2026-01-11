import argparse, json
from pathlib import Path
import logging
from torch.utils.data import DataLoader

from src.config import load_config
from src.utils import  load_processed_np, load_for_evaluate, validate
from src.data_loader import ChurnDataset

logger = logging.getLogger(__name__)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    processed_dir = Path(cfg["data"].get("processed_dir", "data/processed"))
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    model, preproc, device = load_for_evaluate(cfg, logger)
    X_train, y_train, X_val, y_val = load_processed_np(processed_dir)
    val_dataset = ChurnDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"])
    metrics = validate(model, val_loader, device, cfg)

    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()