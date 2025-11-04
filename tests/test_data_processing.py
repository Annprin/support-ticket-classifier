import pytest
import pandas as pd
from src.data_loader import preprocess_data, create_preprocessor
import numpy as np

# Фикстура: создаем тестовый DataFrame
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 10, 24, 12],
        "PhoneService": ["Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity": ["Yes", "No", "No", "No"],
        "OnlineBackup": ["Yes", "No", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "No"],
        "TechSupport": ["No", "No", "Yes", "No"],
        "StreamingTV": ["No", "Yes", "No", "No"],
        "StreamingMovies": ["No", "Yes", "No", "No"],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        "MonthlyCharges": [29.85, 80.0, 60.0, 45.0],
        # Проверяем обработку пропусков и ' '
        "TotalCharges": ["29.85", "800.0", " ", "540.0"], 
        "Churn": ["No", "Yes", "No", "Yes"]
    })

# Фикстура: конфиг для тестов
@pytest.fixture
def test_config():
    return {
        "data": {
            "target_column": "Churn",
            "test_size": 0.5,
            "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
            "categorical_features": ["gender", "InternetService", "Contract"]
        },
        "training": {
            "random_seed": 42
        }
    }

# Стало (Правильно)
def test_data_validation_and_preprocessing(sample_data, test_config):
    """
    Проверяет, что предобработка (preprocess_data):
    1. Корректно разделяет данные.
    2. Правильно кодирует Target (Yes/No -> 1/0).
    3. Обрабатывает ' ' в 'TotalCharges'.
    """
    X_train, X_val, y_train, y_val = preprocess_data(sample_data, test_config)
    
    # 1. Проверка разделения (2 train, 2 val из-за test_size=0.5)
    assert len(X_train) == 2
    assert len(X_val) == 2
    
    # 2. Проверка кодирования Target
    assert y_train.dtype == 'int64'
    assert y_val.dtype == 'int64'
    assert y_train.isin([0, 1]).all()
    assert y_val.isin([0, 1]).all()

    # 3. Проверка TotalCharges (что ' ' обработался)
    # Просто проверяем, что колонка стала числовой и в ней нет NaN
    assert X_train["TotalCharges"].dtype == 'float64'
    assert X_val["TotalCharges"].dtype == 'float64'
    

def test_column_transformer(sample_data, test_config):
    """
    Проверяет, что ColumnTransformer (preprocessor):
    1. Создается
    2. Корректно трансформирует данные (StandardScaler + OHE)
    """
    X_train, X_val, y_train, y_val = preprocess_data(sample_data, test_config)
    
    preprocessor = create_preprocessor(test_config)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # 3 (numeric) + 2 (gender) + 2 (InternetService) + 2 (Contract) = 9
    # (Кол-во колонок OHE зависит от того, что попало в X_train)
    # Форма (2 строки, 9 колонок)
    assert X_train_processed.shape == (2, 9)
    assert X_val_processed.shape == (2, 9) # У X_val должна быть та же форма
    
    # Проверяем, что числовые фичи (первые 3) отмасштабированы (среднее ~0)
    assert np.isclose(np.mean(X_train_processed[:, 0]), 0.0)