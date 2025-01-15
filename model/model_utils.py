import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple


class YOLOv8Segmenter:
    """
    Класс для работы с моделью YOLOv8 для сегментации объектов, выделяя только класс "собака".
    """
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        Инициализация модели YOLOv8.

        :param model_path: Путь к файлу модели ONNX.
        :param input_size: Размер для изменения входного изображения (ширина, высота).
        """
        self.model_path = model_path
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path)

    def preprocess_image(self, image_path: str):
        """
        Загружает изображение, изменяет размер и нормализует его для модели.

        :param image_path: Путь к изображению.
        :return: Нормализованное изображение, готовое для подачи в модель.
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Изменяем размер и нормализуем изображение
        image_resized = cv2.resize(image, self.input_size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb / 255.0

        # Меняем размерность для подачи в модель (Batch size, H, W, C)
        image_input = np.expand_dims(image_normalized, axis=0).astype(np.float32)

        return image_input, image

    def get_dog_mask(self, image_input: np.ndarray):
        """
        Получает маску для класса "собака" из результата инференса модели.

        :param image_input: Входное изображение для инференса.
        :return: Маска для класса "собака".
        """
        # Подаем изображение в модель
        inputs = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()[0].name
        result = self.session.run([outputs], {inputs: image_input})[0]

        # Разделяем результат на классы и маски
        # Примерный формат результата: [batch, num_boxes, 4 (box), 1 (mask), num_classes]
        boxes, masks, class_ids = result[..., :4], result[..., 4], result[..., 5]

        # Фильтруем только маски для класса "собака" (ID для "dog" обычно 16 для COCO)
        dog_mask = np.zeros_like(masks[0], dtype=np.uint8)
        dog_class_id = 16  # ID класса "собака" в COCO

        for i in range(masks.shape[1]):
            if class_ids[0][i] == dog_class_id:
                dog_mask += masks[0][i]

        # Применяем порог, чтобы получить бинарную маску
        dog_mask = (dog_mask > 0.5).astype(np.uint8) * 255

        return dog_mask

    def apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray):
        """
        Применяет маску к изображению, вырезая собаку и делая остальной фон прозрачным.

        :param image: Оригинальное изображение.
        :param mask: Маска для собаки.
        :return: Изображение с вырезанным объектом.
        """
        # Создаем альфа-канал
        alpha_channel = mask.astype(np.uint8)

        # Объединяем RGB изображение с альфа-каналом
        image_rgba = np.dstack((image, alpha_channel))

        return image_rgba

    def segment_dog(self, image_path: str, output_path: str):
        """
        Основной метод для выполнения сегментации собаки на изображении.

        :param image_path: Путь к изображению.
        :param output_path: Путь для сохранения результата.
        """
        # Подготовка изображения
        image_input, original_image = self.preprocess_image(image_path)

        # Получение маски для собаки
        dog_mask = self.get_dog_mask(image_input)

        # Применяем маску к изображению
        result_image = self.apply_mask_to_image(original_image, dog_mask)

        # Сохраняем итоговое изображение
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGRA))
