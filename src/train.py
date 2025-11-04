import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import joblib

from . import data_loader, model as model_loader, utils
from .model import ChurnModelConfig, ChurnMLP

logger = logging.getLogger(__name__)

def train_model(config: dict, verbose: bool = True):
    """Основная функция для запуска обучения."""
    
    # 1. Установка seed
    utils.set_seed(config['training']['random_seed'])
    logger.info("Random seed установлен.")
    
    # 2. Загрузка и обработка данных
    df_raw = data_loader.load_data(config['data']['path'])
    X_train_df, X_val_df, y_train, y_val = data_loader.preprocess_data(df_raw, config)
    
    # 3. Создание и обучение препроцессора (ColumnTransformer)
    preprocessor = data_loader.create_preprocessor(config)
    
    # Обучаем препроцессор ТОЛЬКО на X_train
    X_train_processed = preprocessor.fit_transform(X_train_df)
    # Применяем препроцессор к X_val
    X_val_processed = preprocessor.transform(X_val_df)
    
    # Сохраняем препроцессор (Задание 2)
    joblib.dump(preprocessor, config['model']['preprocessor_path'])
    logger.info(f"Препроцессор сохранен в {config['model']['preprocessor_path']}")
    
    # Получаем кол-во фичей после OHE для конфига модели
    input_features_count = X_train_processed.shape[1]
    logger.info(f"Количество признаков после OHE: {input_features_count}")
    
    # 4. Создание Datasets и DataLoaders
    train_dataset = data_loader.ChurnDataset(X_train_processed, y_train.values)
    val_dataset = data_loader.ChurnDataset(X_val_processed, y_val.values)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    # 5. Инициализация модели (по стандартам HF)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Обучение на устройстве: {device}")

    # Создаем конфиг НС из config.yaml
    model_config = ChurnModelConfig(
        input_size=input_features_count,
        hidden_layers=config['architecture']['hidden_layers'],
        output_size=len(config['labels_map']), # 2
        dropout=config['architecture']['dropout']
    )
    
    # Создаем саму модель
    model = ChurnMLP(model_config)
    model.to(device)
    
    # 6. Оптимизатор и планировщик
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    num_training_steps = config['training']['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    # 7. Цикл обучения
    logger.info("...::: Начало обучения :::...")
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", disable=not verbose)
        
        for batch in progress_bar:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(input_features=features, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # 8. Валидация (Задание 2: вывод метрик в консоль)
        val_metrics = validate(model, val_loader, device, config)
        logger.info(f"Epoch {epoch+1} | Validation | Accuracy: {val_metrics['accuracy']:.4f} | "
                    f"F1 (Churn): {val_metrics['f1_churn']:.4f} | "
                    f"Precision (Churn): {val_metrics['precision_churn']:.4f}")

    logger.info("...::: Обучение завершено :::...")
    
    # 9. Сохранение модели (Задание 2: save_pretrained)
    save_path = config['model']['save_path']
    model.save_pretrained(save_path)
    logger.info(f"Модель сохранена в {save_path} (формат Hugging Face)")

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