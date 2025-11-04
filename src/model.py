import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import logging

logger = logging.getLogger(__name__)

# 1. Создаем кастомный Config, наследуемый от PretrainedConfig
class ChurnModelConfig(PretrainedConfig):
    model_type = "ChurnMLP"
    
    def __init__(
        self,
        input_size: int = 20,
        hidden_layers: list = [64, 32],
        output_size: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout
        super().__init__(**kwargs)

# 2. Создаем кастомную Модель, наследуемую от PreTrainedModel
class ChurnMLP(PreTrainedModel):
    # Указываем наш класс конфига
    config_class = ChurnModelConfig
    
    def __init__(self, config: ChurnModelConfig):
        super().__init__(config)
        
        layers = []
        in_features = config.input_size
        
        # Динамически строим слои на основе конфига
        for h_dim in config.hidden_layers:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_features = h_dim
            
        # Последний слой (логиты)
        layers.append(nn.Linear(in_features, config.output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Инициализируем веса (требование PreTrainedModel)
        self.post_init()

    def _init_weights(self, module):
        """ Инициализирует веса."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        
        # `input_features` - это X_train
        logits = self.network(input_features)
        
        loss = None
        if labels is not None:
            # Для несбалансированных данных можно добавить class_weights
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.output_size), labels.view(-1))
            
        # Возвращаем стандартный HF Output объект
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )