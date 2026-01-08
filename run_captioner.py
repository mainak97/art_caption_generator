import os
import sys
import json
import torch
from PIL import Image

# ---------- CONFIG ----------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VISION_MODEL = "Salesforce/blip2-opt-2.7b"
TEXT_MODEL = "google/flan-t5-base"
DEVICE = "cpu"
# ----------------------------

def is_image(name):
    return os.path.splitext(name.lower())[1] in IMAGE_EXTS

# ---------- LOAD MODELS ----------
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

print("Loading vision model (BLIP2)...")
vision_processor = Blip2Processor.from_pretrained(VISION_MODEL)
vision_model = Blip2ForConditionalGeneration.from_pretrained(
    VISION_MODEL,
    device_map={"": DEVICE},
    torch_dtype=torch.float32,
)
vision_model.eval()

print("Loading text model (FLAN-T5-BASE)...")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
text_model = AutoModelForSeq2SeqLM.from_pretrained(
    TEXT_MODEL,
    device_map="cpu",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
)
text_model.eval()

print("Models loaded.\n")

# ---------- IMAGE EMBEDDING → CAPTION ----------
@torch.no_grad()
def image_to_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    # Extract BLIP2 image embeddings (bypass literal text)
    inputs = vision_processor(images=image, return_tensors="pt").to(DEVICE)
    image_embeds = vision_model.get_image_features(**inputs)

    # Flatten embeddings to a vector and convert to list
    emb_list = image_embeds[0].tolist()

    # Create prompt using the embedding vector
    prompt = (
        "You are a poet. Generate a single-line artistic caption inspired by this image. "
        "Focus on feeling, mood, color, and movement. Do not describe objects literally.\n\n"
        f"Visual embedding: {emb_list[:512]}..."  # truncate for token limit
    )

    # Tokenize and generate
    inputs_text = text_tokenizer(prompt, return_tensors="pt").to(text_model.device)
    ids = text_model.generate(
        **inputs_text,
        max_new_tokens=30,
        do_sample=True,
        top_p=0.9,
        temperature=1.2,
        repetition_penalty=1.2,
        num_beams=3,
        num_return_sequences=1
    )

    caption = text_tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return caption

# ---------- MAIN ----------
def main(root_dir: str, out_dir: str):
    results = {}

    for dirpath, _, filenames in os.walk(root_dir):
        images = [f for f in filenames if is_image(f)]
        if not images:
            continue

        folder = os.path.basename(dirpath)
        results[folder] = []

        for fname in images:
            path = os.path.join(dirpath, fname)
            print(f"Processing {path}")

            try:
                caption = image_to_caption(path)
            except Exception as e:
                caption = f"ERROR: {e}"

            results[folder].append({
                "filename": fname,
                "caption": caption
            })

    out_path = os.path.join(out_dir, "captions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved captions to {out_path}")

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise RuntimeError("Usage: python run_captioner_fast.py <image_root_folder> <output_root_folder>")

    main(sys.argv[1],sys.argv[2])
