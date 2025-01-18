import torch
from colpali_engine.models import ColPali, ColPaliProcessor

class ColPaliModel:
    def __init__(self, model_path="vidore/colpali-v1.3", device="cuda"):
        self.device = device
        self.model = ColPali.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.processor = ColPaliProcessor.from_pretrained(model_path)

    def embed_text(self, text: str):
        """
        Embeds a given text using ColPali's text pipeline.
        Returns a NumPy array, shape (1, 128).
        """
        batch_queries = self.processor.process_queries([text]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch_queries)
        return embeddings.cpu().numpy()

    def embed_image(self, image):
        """
        Embeds a given PIL image using ColPali's image pipeline.
        Returns a pooled embedding, shape (1, 128).
        """
        batch_images = self.processor.process_images([image]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch_images)
        pooled_embedding = embeddings.mean(dim=1)
        return pooled_embedding.cpu().numpy()