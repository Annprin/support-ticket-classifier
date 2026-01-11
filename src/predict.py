import argparse
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import torch

from src.config import load_config
from src.utils import load_for_evaluate

logger = logging.getLogger(__name__)

def parse_args():
    ap = argparse.ArgumentParser(description="Offline inference for Telco Churn model")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--input_path", required=True, help="Path to input CSV with features")
    ap.add_argument("--output_path", required=True, help="Path to output CSV with predictions")
    ap.add_argument("--proba", action="store_true", help="If set, also write churn probability")
    return ap.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    cfg = load_config(args.config)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input_path not found: {input_path}")

    # Load model + preprocessor
    model, preproc, device = load_for_evaluate(cfg, logger)
    model.eval()

    # Read input
    df = pd.read_csv(input_path)

    # If target accidentally present, drop it
    target_col = cfg.get("data", {}).get("target_col", "Churn")
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Transform features -> matrix
    X = preproc.transform(df)

    # Handle sparse / dataframe to numpy
    try:
        from scipy import sparse
        if sparse.issparse(X):
            X = X.toarray()
    except Exception:
        pass

    if hasattr(X, "to_numpy"):
        X = X.to_numpy()

    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(input_features=X_tensor)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_class = np.argmax(probs, axis=1)

    # Map predicted class -> label
    # config['labels_map'] assumed like {"No": 0, "Yes": 1}
    labels_map = cfg.get("labels_map", {"No": 0, "Yes": 1})
    id2label = {v: k for k, v in labels_map.items()}
    pred_label = [id2label.get(int(i), str(int(i))) for i in pred_class]

    out_df = pd.DataFrame({"prediction": pred_label})
    if args.proba:
        # probability of class "Yes" (churn)
        churn_id = labels_map.get("Yes", 1)
        out_df["proba_yes"] = probs[:, churn_id]

    out_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to: {output_path} (rows={len(out_df)})")

if __name__ == "__main__":
    main()
