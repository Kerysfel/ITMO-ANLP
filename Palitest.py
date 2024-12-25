import requests
from PIL import Image
import io

# 1. Проверка текстового запроса (на случай, если с изображениями будет медленно)
def test_text_embedding():
    API_URL = "http://127.0.0.1:5000/embed/text"  # Подставь свой IP
    data = {
        "queries": ["Hello, how are you?"]
    }
    
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        print("Text embedding работает!")
        print("Ответ сервера:", response.json())
    else:
        print("Ошибка:", response.status_code, response.text)

# 2. Проверка обработки изображения
def test_image_embedding():
    API_URL = "http://127.0.0.1:5000/embed/image"  # Подставь свой IP
    image_path = "test.png"  # Подставь путь к изображению (PNG/JPG)
    
    # Открываем изображение и отправляем его
    with open(image_path, "rb") as f:
        files = {'file': f}
        response = requests.post(API_URL, files=files)
    
    if response.status_code == 200:
        print("Image embedding работает!")
        print("Ответ сервера:", response.json())
    else:
        print("Ошибка:", response.status_code, response.text)

# Запускаем тесты
test_text_embedding()
test_image_embedding()
