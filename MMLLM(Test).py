import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Убедитесь, что библиотека qwen-vl-utils установлена:
# pip install qwen-vl-utils

# 1. Загрузка модели с опциональной 4-битной квантизацией
# Если у вас ограниченная видеопамять, можно добавить load_in_4bit=True.
# Если VRAM хватает, можно убрать load_in_4bit или использовать bf16.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # Оставляем квантизацию в 4 бита
)

# 2. Загрузка процессора
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 3. Опционально зададим min/max пикселей, если нужно ограничение по VRAM
min_pixels = 256 * 14 * 14   # = 256 * 196
max_pixels = 512 * 14 * 14   # = 512 * 196
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)

# 4. Путь к изображению (локальный файл). Подставьте нужный путь.
image_path = "B:/ITMO/NLPCSprog/test.png"
file_uri = f"file://{image_path}"

# 5. Формируем сообщения.
# Обратите внимание, что model “думает”, будто получает ChatGPT-стиль сообщения.
# Внутри "content" мы передаём список словарей [{"type": ..., "image": ...}, {"type": ..., "text": ...}].
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": file_uri
            },
            {
                "type": "text",
                "text": "What are the types of neural networks mentioned in this image?"
            }
        ],
    }
]

# 6. Генерируем текстовое представление (prompt) для модели
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 7. Обрабатываем само изображение
image_inputs, video_inputs = process_vision_info(messages)

# 8. Преобразуем всё в нужный формат для модели (BatchEncoding)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# 9. Переносим тензоры на CUDA (если есть GPU)
inputs = inputs.to("cuda")

# 10. Вызываем модель для генерации ответа
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 11. Опционально “обрезаем” исходную часть input_ids (если нужно исключить prompt)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 12. Декодируем токены в текст
output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)

# 13. Вывод ответа
print("Ответ модели:", output_text[0])
