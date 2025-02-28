import os

input_folder = 'V:/NP/dataset/labels'  # Папка с исходными файлами `.chunk`
output_folder = 'V:/NP/dataset/labels_txt'  # Папка для файлов `.txt`

# Размеры изображений (замените на реальные размеры ваших изображений)
image_width = 1920
image_height = 1200

# Словарь для сопоставления названий классов с class_id
class_mapping = {
    "Cargo_Bulker_4": 0,
    "MISV_Yury_Topchev": 0,
    "Cargo_Bulker_5": 0,
    "Cargo_RiverSea_8": 0,
    "Passenger_RiverSeaCruiseShip_1": 1,
    "Passenger_RiverCruiseShip_1": 1,
    "Passenger_CruiseShip_2": 1,
    "Tug_ASD_Ice_11": 2,
    "Pushboat_Conventional_3": 2,
    "Tug_ATD_3": 2,
    "Light_SpeedBoat_2": 3,
    "Light_SpeedBoat": 3,
    "Light": 3,
    "Light_RIB_1": 3,
    "Small_craft_2": 3,
    "Navy_PatrolShip_1": 4,
    "Navy_Frigate_1": 4,
    "Navy_Frigate_7": 4,
    "Submarine_Kilo_class_pr_877": 5,
    "Submarine_Oskar_II_class_pr_949A": 5,
    "KZ_BuoyConical": 6,
    "YachtBY": 6,
    "esKhPG_PS": 6,
    "esKhPG_SB": 6
}

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.chunk'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.chunk', '.txt'))

        # Ищем название класса в имени файла
        class_name = None
        for key in class_mapping.keys():
            if key in filename:  # Если ключ из словаря найден в имени файла
                class_name = key
                break

        if class_name is None:
            print(f"Skipping {filename}: Class name not found in filename.")
            continue

        # Получаем class_id из словаря
        class_id = class_mapping[class_name]

        with open(input_path, 'r') as chunk_file:
            lines = chunk_file.readlines()
            if len(lines) > 1:  # Проверяем, что файл не пуст
                try:
                    x_value, y_value, with_value, height_value = None, None, None, None
                    for line in lines:
                        if 'float x' in line:
                            x_value = float(line.split('=')[1].strip().rstrip(';'))
                        elif 'float y' in line:
                            y_value = float(line.split('=')[1].strip().rstrip(';'))
                        elif 'float with' in line:  # Замена 'width' на 'with'
                            with_value = float(line.split('=')[1].strip().rstrip(';'))
                        elif 'float height' in line:
                            height_value = float(line.split('=')[1].strip().rstrip(';'))

                    if x_value is not None and y_value is not None and with_value is not None and height_value is not None:
                        # Обрезаем координаты рамки по границам изображения
                        x = max(0, min(x_value, image_width - 1))
                        y = max(0, min(y_value, image_height - 1))
                        width = max(0, min(with_value, image_width - x))
                        height = max(0, min(height_value, image_height - y))

                        # Нормализуем координаты для YOLO
                        x_center = (x + width / 2) / image_width
                        y_center = (y + height / 2) / image_height
                        width_norm = width / image_width
                        height_norm = height / image_height

                        # Проверяем, что координаты в пределах [0, 1]
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width_norm <= 1 and 0 <= height_norm <= 1:
                            with open(output_path, 'w') as txt_file:
                                txt_file.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
                        else:
                            print(f"Skipping {filename}: Normalized coordinates out of bounds.")
                except (IndexError, ValueError) as e:
                    print(f"Error processing {filename}: {e}. Skipping due to invalid data.")