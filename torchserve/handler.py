import json
import logging
from pathlib import Path

import numpy as np
import torch
import joblib

from ts.torch_handler.base_handler import BaseHandler

from src.model import ChurnMLP, ChurnModelConfig

logger = logging.getLogger(__name__)

class ChurnHandler(BaseHandler):
    """
    Expects JSON:
      - {"instances": [ {<feature>: <value>, ...}, ... ]}
      - or a single dict {<feature>: <value>, ...}
      - or a raw list of dicts [ {...}, {...} ]
    Returns:
      [{"prediction": "Yes"/"No", "proba_yes": float}, ...]
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.preproc = None
        self.id2label = None
        self.yes_id = 1

    def initialize(self, ctx):
        props = ctx.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = Path(props.get("model_dir"))
        manifest = ctx.manifest

        # extra_files are placed into model_dir
        # We will ship: config.json + preprocessor.joblib
        cfg_path = model_dir / "config.json"
        preproc_path = model_dir / "preprocessor.joblib"
        weights_path = model_dir / "model.pt"  # state_dict from torch.save

        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        if not preproc_path.exists():
            raise FileNotFoundError(f"preprocessor.joblib not found in {model_dir}")
        if not weights_path.exists():
            raise FileNotFoundError(f"model.pt not found in {model_dir}")

        cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
        model_cfg = ChurnModelConfig(
            input_size=cfg_json["input_size"],
            hidden_layers=cfg_json["hidden_layers"],
            output_size=cfg_json["output_size"],
            dropout=cfg_json["dropout"],
        )

        self.model = ChurnMLP(model_cfg)
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.preproc = joblib.load(preproc_path)

        # labels map (fallback)
        self.id2label = {0: "No", 1: "Yes"}
        self.yes_id = 1

        logger.info(f"Initialized model from {model_dir} on device {self.device}")

    def preprocess(self, data):
        # TorchServe gives list of requests; take first
        raw = data[0].get("data") or data[0].get("body")
        if raw is None:
            raise ValueError("Empty request body")

        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")

        if isinstance(raw, str):
            payload = json.loads(raw)
        else:
            payload = raw

        # Normalize payload -> list[dict]
        if isinstance(payload, dict) and "instances" in payload:
            instances = payload["instances"]
        elif isinstance(payload, dict):
            instances = [payload]
        elif isinstance(payload, list):
            instances = payload
        else:
            raise ValueError("Unsupported JSON format")

        # to DataFrame-like: sklearn transformer accepts list[dict] via pandas DataFrame
        import pandas as pd
        df = pd.DataFrame(instances)

        X = self.preproc.transform(df)

        # sparse -> dense
        try:
            from scipy import sparse
            if sparse.issparse(X):
                X = X.toarray()
        except Exception:
            pass

        X = np.asarray(X, dtype=np.float32)
        return torch.tensor(X, dtype=torch.float32)

    def inference(self, inputs):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model(input_features=inputs)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        return preds, probs

    def postprocess(self, inference_output):
        preds, probs = inference_output
        result = []
        for i, cls_id in enumerate(preds):
            cls_id = int(cls_id)
            result.append(
                {
                    "prediction": self.id2label.get(cls_id, str(cls_id)),
                    "proba_yes": float(probs[i, self.yes_id]),
                }
            )
        return [result]
