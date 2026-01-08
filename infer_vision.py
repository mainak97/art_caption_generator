import sys
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

MODEL_ID = "Salesforce/blip2-opt-2.7b"
DEVICE = "cpu"


def sanitize_semantic(text: str) -> str:
    text = text.strip()

    # Remove common leading patterns
    prefixes = [
        "mood and atmosphere:",
        "atmosphere:",
        "mood:",
        "the mood is",
        "the atmosphere is",
    ]

    lower = text.lower()
    for p in prefixes:
        if lower.startswith(p):
            text = text[len(p):].strip()

    # Drop instruction-like lines entirely
    banned = [
        "describe",
        "avoid",
        "rewrite",
        "caption",
        "instruction",
    ]

    lines = [
        l for l in text.splitlines()
        if not any(b in l.lower() for b in banned)
    ]

    return " ".join(lines).strip()


def main(image_path: str):
    processor = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map={"": DEVICE},
        torch_dtype=torch.float32,
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")

    # Completion-style prompt, NOT instruction
    prompt = ""

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            num_beams=3,
        )

    raw = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
    clean = sanitize_semantic(raw)

    print(clean)


if __name__ == "__main__":
    main(sys.argv[1])
