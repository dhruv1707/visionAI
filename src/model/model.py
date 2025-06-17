import torch
import torch.nn as nn
from PIL import Image
import requests
from transformers import AutoProcessor, MllamaForConditionalGeneration

class Model(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
    
    def forward(self, img, prompt):
        inputs = self.processor(img, prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=30)
        print(self.processor.decode(output[0]))

if __name__=="__main__":
    model_id = "meta-llama/Llama-3.2-11B-Vision"
    model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,)

    processor = AutoProcessor.from_pretrained(model_id)
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

    image = Image.open(requests.get(url, stream=True).raw) 
    prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
    model = Model(model, processor)
    model(image, prompt)

