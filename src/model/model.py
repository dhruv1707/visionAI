# import torch
# import torch.nn as nn
# from PIL import Image
# import requests
# from transformers import AutoProcessor, MllamaForConditionalGeneration

# class Model(nn.Module):
#     def __init__(self, model, processor):
#         super().__init__()
#         self.model = model
#         self.processor = processor
    
#     def forward(self, img, prompt, transcript):
#         inputs = self.processor(images = img, text = prompt, return_tensors="pt")
#         print(self.processor)
#         print(f"Inputs: {inputs}")
#         print(inputs['input_ids'].shape)
#         print(inputs['pixel_values'].shape)

#         output = self.model.generate(**inputs, max_new_tokens=30)
#         print(f"Output: {output}")
#         print(self.processor.decode(output[0]))


# if __name__=="__main__":
#     model_id = "meta-llama/Llama-3.2-11B-Vision"
    
#     model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True)

#     processor = AutoProcessor.from_pretrained(model_id)
#     url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

#     image = Image.open(requests.get(url, stream=True).raw) 
#     transcript = "If you people have any problem, just come to me. Whether you need money or something else, just ask! Brother!"
#     prompt = (
#               "<|begin_of_text|>"
#                 "<|start_header_id|>user<|end_header_id|>"
#                 "<|image|> "
#                 "Describe what’s happening in this frame and in the transcript below:\n\n"
#                 f"{transcript}\n\n"
#               "<|eot_id|>"
#                 "<|start_header_id|>assistant<|end_header_id|>"
#               "<|end_of_text|>"
#             )
#     model = Model(model, processor)
#     model(image, prompt, transcript)
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, model_1, processor_1):
        super().__init__()
        self.model_1 = model_1
        self.processor_1 = processor_1
        

    def forward(self, path_to_data):
      images = []
      summaries = []
      for dirpath, dirnames, filenames in os.walk(path_to_data):
        print(f"Dirpath: {dirpath}")
        print(f"Dirname: {dirnames}")
        for file in filenames:
          if file.endswith(".txt"):
            transcript_path = os.path.join(dirpath, file)
            with open(transcript_path, 'r', encoding="utf-8") as f:
              transcript = f.read().strip()

        for file in filenames:
          if not file.endswith(".txt"):

            image_path = os.path.join(dirpath, file)
            image = Image.open(image_path)

            # prompt = "Please summarize the actions taking place in this frame. Be as descriptive as possible."

            inputs = self.processor_1(images = [image],
                                  return_tensors="pt").to(device)

            print(f"Inputs: {inputs}")
            # print(f"Input IDs shape: {inputs['input_ids'].shape}")
            print(f"Pixel values shape: {inputs['pixel_values'].shape}")


            output = self.model_1.generate(**inputs, max_new_tokens=50,
                                         do_sample=True, top_p=0.9,
                                         temperature=0.7,)
            summaries.append(self.processor_1.decode(output[0], skip_special_tokens=True))

            print(f"Output: {self.processor_1.decode(output[0], skip_special_tokens=True)}")
            print(summaries)
    


if __name__=="__main__":

  model_id_image = "Salesforce/blip2-opt-2.7b"
  processor_1 = Blip2Processor.from_pretrained(model_id_image)
  model_1 = Blip2ForConditionalGeneration.from_pretrained(model_id_image,
                                                        device_map="auto")
  tokenizer = processor_1.tokenizer
  token = tokenizer.convert_ids_to_tokens(50265)
  print("50265 →", token)


  path_to_data = "./data"

  model = Model(model_1, processor_1)
  model(path_to_data)
