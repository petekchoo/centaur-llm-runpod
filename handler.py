from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

model_path = "/root/centaur_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    print(f"⏱️ Took {time.time() - start:.2f} sec")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(generate("What is the nature of intelligence?"))