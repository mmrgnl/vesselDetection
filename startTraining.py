from ultralytics import YOLO
import os
import torch

torch.cuda.empty_cache()

# Установка устройства
device = torch.device('cuda')
#print(f"Using device: {device}")

# Настройка параметров
YAML_FILE = "V:/NP/data.yaml"  # Путь к YAML файлу
MODEL_PATH = "V:/NP/mixShipsV5.pt"  # Базовая модель YOLOv11
OUTPUT_PATH = "V:/clear/models/Yury_Topchev_v1.pt"  # Путь для сохранения обученной модели
EPOCHS = 11  # Количество эпох
IMG_SIZE = (1920, 1200)  # Размер изображения (можно изменить на 640, 1280, 1920 и т.д. в зависимости от потребностей)
BATCH_SIZE = 4 # Размер батча
DEVICE = device  # Используем CUDA или CPU

# Проверка на наличие YAML файла
if not os.path.exists(YAML_FILE):
    raise FileNotFoundError(f"YAML файл не найден по пути: {YAML_FILE}")

def train_model():
    print("Запуск обучения...")
    model = YOLO(MODEL_PATH)  # Загрузка базовой модели YOLOv11

    # Запуск процесса обучения
    model.train(
        data=YAML_FILE,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs/train",
        name="exp",
        verbose=True, # Информация о ходе обучения будет отображаться
        #resume=True
    )

    # Сохранение моделии
    model.export(path=OUTPUT_PATH)
    print(f"Обучение завершено. Модель сохранена в {OUTPUT_PATH}")

if __name__ == "__main__":  # Обязательно добавьте эту конструкцию
    train_model()
