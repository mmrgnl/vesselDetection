import os
import shutil
from sklearn.model_selection import train_test_split

# Путь к основной папке FistNP
base_dir = 'V:/NP'

# Путь к папкам с изображениями и метками
image_dir = 'V:/NP/dataset/images'
labels_dir = 'V:/NP/dataset/labels_txt'

# Считываем все файлы из директорий
all_images = sorted(os.listdir(image_dir))
all_labels = sorted(os.listdir(labels_dir))

# Создаем списки с парами изображений и меток
image_label_pairs = []
for image in all_images:
    label = os.path.splitext(image)[0] + '.txt'
    if label in all_labels:
        image_label_pairs.append((image, label))

# Разделение данных: 50% train, 30% valid, 20% test
train_size = 0.6
valid_size = 0.3
test_size = 0.1

train_pairs, temp_pairs = train_test_split(image_label_pairs, test_size=(valid_size + test_size), random_state=42)
valid_pairs, test_pairs = train_test_split(temp_pairs, test_size=test_size / (valid_size + test_size), random_state=42)

# Создание структуры директорий
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# Перемещение файлов в соответствующие директории
def move_files(pairs, split):
    for image, label in pairs:
        # Папки назначения
        image_dest = os.path.join(base_dir, split, 'images', image)
        label_dest = os.path.join(base_dir, split, 'labels', label)

        # Перемещение
        shutil.move(os.path.join(image_dir, image), image_dest)
        shutil.move(os.path.join(labels_dir, label), label_dest)

move_files(train_pairs, 'train')
move_files(valid_pairs, 'valid')
move_files(test_pairs, 'test')

print("Файлы успешно распределены по новой структуре!")
