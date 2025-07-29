from runpod.serverless.utils.rp_validator import validate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model on cold start
model_id = "myshell-ai/Centaur-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

@validate
def handler(event):
    prompt = event["input"].get("prompt")
    if not prompt:
        return {"error": "Missing 'prompt' in input"}

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": result}