import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "./text_lora"  # optional

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

try:
    model = PeftModel.from_pretrained(model, LORA_PATH)
except Exception:
    pass

model.eval()

semantic = sys.argv[1]

prompt = (
    "Write a short artistic caption inspired by this:\n\n"
    f"{semantic}\n\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        num_beams=3,
        repetition_penalty=1.2,
    )

print(tokenizer.decode(ids[0], skip_special_tokens=True))
