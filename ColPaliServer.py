from typing import cast
import torch
from PIL import Image
from flask import Flask, request, jsonify
from colpali_engine.models import ColPali, ColPaliProcessor

app = Flask(__name__)

# Загрузка модели
model_name = "vidore/colpali-v1.3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# API для обработки текста
@app.route('/embed/text', methods=['POST'])
def embed_text():
    data = request.json
    queries = data.get("queries", [])
    if not queries:
        return jsonify({"error": "No queries provided"}), 400
    
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    return jsonify({"embeddings": query_embeddings.tolist()})

# API для обработки изображений
@app.route('/embed/image', methods=['POST'])
def embed_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = Image.open(file)

    batch_images = processor.process_images([image]).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)

    return jsonify({"embedding": image_embeddings.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)