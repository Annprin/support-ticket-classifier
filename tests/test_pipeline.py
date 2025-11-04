import pytest
import torch
from src.model import ChurnModelConfig, ChurnMLP
from src.utils import post_process_output
from typing import Dict

@pytest.fixture
def test_model_config():
    # Конфиг для тестовой модели
    return ChurnModelConfig(
        input_size=10, # 10 фичей на входе
        hidden_layers=[8], # 1 скрытый слой
        output_size=2, # 2 класса (No/Yes)
        dropout=0.1
    )

def test_model_creation_and_forward(test_model_config):
    """
    Проверяет, что модель:
    1. Корректно создается с кастомным конфигом.
    2. Выполняет 'forward' и возвращает правильную форму (shape).
    """
    model = ChurnMLP(test_model_config)
    
    # Создаем батч из 4-х примеров
    dummy_input = torch.randn(4, test_model_config.input_size)
    
    # Вызов forward
    outputs = model(input_features=dummy_input)
    
    # Проверяем logits
    assert outputs.logits.shape == (4, 2)
    # Проверяем, что loss нет, т.к. не передали labels
    assert outputs.loss is None

def test_model_training_step(test_model_config):
    """Проверяет, что модель может выполнить шаг обучения (включая loss)."""
    model = ChurnMLP(test_model_config)
    
    dummy_input = torch.randn(4, test_model_config.input_size)
    dummy_labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    
    # Вызов forward с labels
    outputs = model(input_features=dummy_input, labels=dummy_labels)
    
    # Loss должен быть посчитан
    assert outputs.loss is not None
    assert outputs.loss > 0

def test_api_output_processing():
    """
    Проверяет корректность перевода сырых логитов модели
    в формат API (Задание 3).
    """
    # Карта для перевода ID в метку
    id_to_label_map: Dict[int, str] = {0: "No", 1: "Yes"}

    # Случай 1: Уверенность в "No" (Класс 0)
    # Логиты [3.0, -1.0] -> Softmax ~[0.98, 0.02]
    logits_no = torch.tensor([[3.0, -1.0]])
    result_no = post_process_output(logits_no, id_to_label_map)
    
    assert result_no["predicted_label"] == "No"
    assert result_no["predicted_id"] == 0
    assert result_no["confidence"] > 0.95
    assert result_no["churn_probability"] < 0.05 # Вероятность "Yes" (индекс 1)
    
    # Случай 2: Уверенность в "Yes" (Класс 1)
    # Логиты [-2.0, 2.0] -> Softmax ~[0.018, 0.982]
    logits_yes = torch.tensor([[-2.0, 2.0]])
    result_yes = post_process_output(logits_yes, id_to_label_map)
    
    assert result_yes["predicted_label"] == "Yes"
    assert result_yes["predicted_id"] == 1
    assert result_yes["confidence"] > 0.95 # Уверенность в "Yes"
    assert result_yes["churn_probability"] > 0.95 # Вероятность "Yes"