# Artistic Image Captioning (BLIP-2 + LoRA)

Local system for generating **short, artistic, non-literal captions** from `.webp` images.

This repository adapts **language style**, not visual perception.  
Vision remains frozen. Only the linguistic response is reshaped.

## What this does

- Input: `.webp` image  
- Output: one-sentence artistic caption  
- Style: abstract, metaphorical, restrained  
- Execution: fully local after first model download  

This is **not** an object captioner. Literal descriptions are intentionally suppressed.

## Core idea

BLIP-2 already understands images.  
The problem is *how it speaks about them*.

This repo uses **LoRA (Low-Rank Adaptation)** to bias the language model toward artistic expression without retraining vision.

## Architecture

