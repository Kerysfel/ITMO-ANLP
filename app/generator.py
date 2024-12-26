from app.models.qwen2_vl import Qwen2VLModel

qwen2_vl = Qwen2VLModel()

def generate_response(context, query):
    return qwen2_vl.generate(context, query)