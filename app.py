import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil
from pathlib import Path
from config import MODEL_PATH, UPLOAD_DIR, meow
from model.model_utils import YOLOv8Segmenter
from utils import image_utils


print(meow)

app = FastAPI()

# Создаем папку для хранения изображений
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Инициализируем сегментатор
segmenter = YOLOv8Segmenter(model_path=MODEL_PATH)

# Функция для сохранения изображения
def save_image(file: UploadFile):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_location

# Эндпоинт для загрузки изображения и обработки
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_path = save_image(file)

    # Загружаем изображение и делаем предобработку
    image_input, original_image = segmenter.preprocess_image(image_path)

    # Получаем маску для собаки
    dog_mask = segmenter.get_dog_mask(image_input)

    # Применяем маску к изображению
    image_processor = ImageProcessor(target_size=(640, 640))
    image_processor.load_image(image_path)
    image_processor.apply_mask(dog_mask)

    # Сохраняем итоговое изображение
    output_image_path = os.path.join(UPLOAD_DIR, f"result_{file.filename}")
    image_processor.save_image(output_image_path)

    # Отправляем результат обратно
    return FileResponse(output_image_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)