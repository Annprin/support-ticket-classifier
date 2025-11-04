import yaml
import logging
from typing import Dict, Any

# Настройка базового логгера
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурационный файл YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Конфигурация успешно загружена из {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Файл конфигурации не найден: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при чтении конфига: {e}")
        raise