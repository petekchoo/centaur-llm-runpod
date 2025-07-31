from unsloth import FastLanguageModel
import torch

print("🐍 handler.py started")

base_model = "meta-llama/Llama-2-70b-hf"
adapter_name = "marcelbinz/Llama-3.1-Centaur-70B-adapter"

print("🚀 Loading base + adapter")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model,
    adapter_name = adapter_name,
    max_seq_length = 32768,
    dtype = torch.bfloat16,     # or float16
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

print("✅ Centaur adapter loaded on top of base model")

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
    prompt = "Explain how gravity works in space."
    print("📝 Prompt:", prompt)
    print("🧠 Output:", generate(prompt))