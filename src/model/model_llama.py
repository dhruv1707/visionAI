from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import torch.nn as nn

class ModelLlama(nn.Module):
    def __init__(self):
        super().__init__()
        model_id_summarizer = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_id_summarizer)
        self.processor = AutoProcessor.from_pretrained(model_id_summarizer)
    
    def forward(self, summaries):

        bullet_list = "\n".join(f"- {s}" for s in summaries)
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are an expert in summarizing a 30-second video segment. You will be given 5 frame descriptions (one every 6 seconds) and the transcript for that 30-second window.\n"
            "Based on the frame descriptions and transcript, produce a concise, coherent summary of what happens during those 30 seconds.\n"
            "If the transcript is not in English, first translate it into English, then use the translated text along with the frame descriptions to create your summary.\n\n"
            f"Here is the bullet-list of frame descriptions and transcript:\n{bullet_list}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )
        inputs = self.processor(text=prompt, return_tensors="pt")
        final = self.model.generate(**inputs, max_new_tokens=150)
        return self.processor.decode(final[0])
    