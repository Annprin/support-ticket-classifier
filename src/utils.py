import torch
import numpy as np
import random
import logging
from typing import Dict

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