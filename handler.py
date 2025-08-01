from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🐍 handler.py started")

model_path = "/root/centaur_model"  # ✅ local path, not HF repo name

print(f"🚀 Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("✅ Model loaded and ready.")

def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "What does it mean to think like a human?"
    print("📝 Prompt:", prompt)
    print("🧠 Output:", generate(prompt))