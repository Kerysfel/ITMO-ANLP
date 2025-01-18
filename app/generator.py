from app.models.qwen2_vl import Qwen2VLModel

qwen2_vl = Qwen2VLModel()

def generate_response(context, query):
    """
    Generates a response by calling Qwen2VL model with the provided context and query.
    """
    return qwen2_vl.generate(context, query)