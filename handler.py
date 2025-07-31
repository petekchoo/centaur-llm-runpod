from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🐍 handler.py started")

model_id = "marcelbinz/Llama-3.1-Centaur-70B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
    trust_remote_code=True
)
print("✅ Model loaded.")

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test run
if __name__ == "__main__":
    prompt = "Tell me a short story about a space explorer."
    print("📝 Prompt:", prompt)
    output = generate(prompt)
    print("🧠 Output:", output)
