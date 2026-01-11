import argparse
from pathlib import Path
import json
import torch
import logging

from src.model import ChurnMLP, ChurnModelConfig

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/churn_mlp_model", help="Folder with HF saved model")
    ap.add_argument("--out", default="torchserve/model.pt", help="Output path for state_dict")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))

    # ChurnModelConfig expects these keys (input_size, hidden_layers, output_size, dropout)
    model_cfg = ChurnModelConfig(
        input_size=cfg_json["input_size"],
        hidden_layers=cfg_json["hidden_layers"],
        output_size=cfg_json["output_size"],
        dropout=cfg_json["dropout"],
    )

    model = ChurnMLP(model_cfg)

    # load weights from HF folder
    # PreTrainedModel uses safetensors typically; easiest is .from_pretrained
    model = ChurnMLP.from_pretrained(str(model_dir), config=model_cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_path)
    logger.info(f"Saved state_dict to: {out_path}")

if __name__ == "__main__":
    main()
