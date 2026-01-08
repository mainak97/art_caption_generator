import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
LR = 2e-4


class TextDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.samples = [json.loads(l) for l in open(path)]
        self.tok = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = f"Turn this into an artistic caption:\n{s['input']}\n\nCaption:"
        text = prompt + s["output"]
        enc = self.tok(text, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora)
model.train()

ds = TextDataset("data.jsonl", tokenizer)
dl = DataLoader(ds, batch_size=1, shuffle=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for batch in dl:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch, labels=batch["input_ids"])
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f"epoch {epoch+1} loss {loss.item():.4f}")

model.save_pretrained("./text_lora")
