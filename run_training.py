import argparse
import logging
from src.config import load_config
from src.train import train_model
from src.utils import configure_logging

# Настраиваем логирование при импорте
configure_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Запуск обучения модели Churn.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к файлу конфигурации (default: config.yaml)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Включить отображение TQDM прогресс-бара"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Запуск процесса обучения с конфигом: {args.config}")
    
    try:
        # 1. Загрузка конфига
        config = load_config(args.config)
        
        # 2. Запуск обучения
        train_model(config, verbose=args.verbose)
        
        logger.info("Процесс обучения успешно завершен.")
        
    except Exception as e:
        logger.error(f"Критическая ошибка в процессе обучения: {e}", exc_info=True)

if __name__ == "__main__":
    main()