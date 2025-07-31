from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🐍 handler.py started")

model_id = "marcelbinz/Llama-3.1-Centaur-70B"

print(f"🚀 Loading model: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,     # or torch.float16 if needed
    device_map="auto",
    trust_remote_code=True
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
    prompt = "Explain how memory works in the human brain."
    print("📝 Prompt:", prompt)
    print("🧠 Output:", generate(prompt))
