from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

class Qwen2VLModel:
    def __init__(self, model_path="Qwen/Qwen2-VL-2B-Instruct", device="cuda"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self, context: str, image=None, max_tokens=256, temperature=0.7):
        """
        Generates text using Qwen2-VL with optional image input.
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": context}]
            }
        ]

        if image:
            messages[0]["content"].append({"type": "image", "image": image})

        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            images=[image] if image else None,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )

        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]