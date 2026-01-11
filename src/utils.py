import torch
import numpy as np
import random
import logging
from typing import Dict
from pathlib import Path
import joblib

from .model import ChurnMLP
from sklearn.metrics import accuracy_score, f1_score, precision_score

def set_seed(seed: int):
    """Устанавливает random seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_logging():
    """Настраивает корневой логгер."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(), # Вывод в консоль
            logging.FileHandler("training.log") # Вывод в файл
        ]
    )

def post_process_output(logits: torch.Tensor, id_to_label_map: Dict[int, str]) -> dict:
    """
    Обрабатывает сырой вывод модели (логиты) в формат для API.
    (Для Задания 3)
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
        
    probabilities = torch.softmax(logits, dim=-1)
    # Предполагаем, что logits имеет форму [num_classes] или [1, num_classes]
    if probabilities.dim() > 1:
        probabilities = probabilities.squeeze(0)
        
    predicted_id = torch.argmax(probabilities).item()
    predicted_label = id_to_label_map[predicted_id]
    confidence = probabilities[predicted_id].item()
    
    return {
        "predicted_label": predicted_label,
        "predicted_id": predicted_id,
        "confidence": confidence,
        "churn_probability": probabilities[1].item() # Вероятность класса 1 (Отток)
    }

def save_processed_np(out_dir: str | Path, X_train, y_train, X_val, y_val) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)

def load_processed_np(processed_dir: str | Path):
    processed_dir = Path(processed_dir)
    X_train = np.load(processed_dir / "X_train.npy", allow_pickle=False)
    y_train = np.load(processed_dir / "y_train.npy", allow_pickle=False)
    X_val = np.load(processed_dir / "X_val.npy", allow_pickle=False)
    y_val = np.load(processed_dir / "y_val.npy", allow_pickle=False)
    return X_train, y_train, X_val, y_val

def load_for_evaluate(config: dict, logger: logging.Logger):
    """
    Ожидает:
      config['model']['save_path'] -> папка с config.json + model.safetensors
      config['model']['preprocessor_path'] -> путь к preprocessor.joblib
    Возвращает: model, preprocessor, device
    """
    model_dir = Path(config["model"].get("save_path", "models/churn_mlp_model"))
    preproc_path = Path(config["model"].get("preprocessor_path", "models/churn_mlp_model/preprocessor.joblib"))

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preproc_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Preprocessor
    preprocessor = joblib.load(preproc_path)
    logger.info(f"Loaded preprocessor: {preproc_path}")

    # 2) HF-model (config.json + model.safetensors)
    model = ChurnMLP.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from: {model_dir} on device: {device}")

    return model, preprocessor, device

def validate(model, val_loader, device, config):
    """Базовая валидация модели."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch
            features = features.to(device)
            
            outputs = model(input_features=features)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Метрики
    # pos_label=1, т.к. "Yes" (Churn) = 1
    f1_churn = f1_score(all_labels, all_preds, pos_label=1)
    precision_churn = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_churn": f1_churn,
        "precision_churn": precision_churn
    }