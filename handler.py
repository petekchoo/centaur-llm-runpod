from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import sys

print("🐍 handler.py started")

model_path = "/root/centaur_model"  # ✅ local path, not HF repo name

print(f"🚀 Loading model from: {model_path}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # ✅ efficient on A100
    device_map="auto",
    low_cpu_mem_usage=True
)

print("✅ Model loaded and ready.")

def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("🧠 Generating output...")
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    end = time.time()
    print(f"⏱️ Generation took {end - start:.2f} seconds")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "What does it mean to think like a human?"

    print("📝 Prompt:", prompt)
    output = generate(prompt)
    print("🧠 Output:", output)