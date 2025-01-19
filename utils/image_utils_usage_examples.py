from utils.image_utils import ImageProcessor
import numpy as np

# Создаем объект процессора
processor = ImageProcessor(target_size=(128, 128))

# Загружаем изображение
processor.load_image("input.jpg")

# Масштабируем изображение
processor.preprocess()

# Создаем пример маски (объект в центре, фон вокруг)
mask = np.zeros((128, 128), dtype=np.uint8)
mask[32:96, 32:96] = 1  # Пример объекта

# Применяем маску
processor.apply_mask(mask)

# Сохраняем результат
processor.save_image("output.png")
