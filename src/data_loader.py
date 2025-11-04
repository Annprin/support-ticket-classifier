import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, Any, Tuple
import joblib

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Загружает данные из CSV."""
    logger.info(f"Загрузка данных из {path}")
    return pd.read_csv(path)

def preprocess_data(
    df: pd.DataFrame, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Выполняет базовую очистку и разделение данных."""
    
    # 1. Очистка
    # TotalCharges - это object, ' ' должны быть NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    # Заполним пропуски (например, медианой)
    median_charges = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_charges)
    logger.info(f"TotalCharges обработаны, пропуски заполнены {median_charges}")

    # 2. Кодирование целевой переменной
    target_col = config['data']['target_column']
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    # 0 = No, 1 = Yes
    
    # 3. Разделение
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['training']['random_seed'],
        stratify=y # Важно для несбалансированных данных
    )
    
    logger.info(f"Данные разделены: Train {X_train.shape[0]}, Val {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val

def create_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """Создает ColumnTransformer для числовых и категориальных признаков."""
    
    numeric_features = config['data']['numeric_features']
    categorical_features = config['data']['categorical_features']
    
    # Пайплайн для числовых признаков
    numeric_transformer = StandardScaler()

    # Пайплайн для категориальных признаков
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Удаляем остальные колонки (если есть)
    )
    return preprocessor

class ChurnDataset(Dataset):
    """Кастомный PyTorch Dataset."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]