from unsloth import FastLanguageModel
import torch

print("🐍 handler.py started")

model_id = "marcelbinz/Llama-3.1-Centaur-70B-adapter"

# Load tokenizer + model using Unsloth
print("🚀 Loading model from:", model_id)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 32768,
    dtype = torch.bfloat16,
    load_in_4bit = False   # ← disables 4-bit & Triton/bitsandbytes
)

FastLanguageModel.for_inference(model)  # Prepare for generation
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

# Test
if __name__ == "__main__":
    prompt = "Explain how gravity works in space."
    print("📝 Prompt:", prompt)
    output = generate(prompt)
    print("🧠 Output:", output)
