import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.handler import runpod_handler

# ─────────────────────────────────────
# Load model at cold start
# ─────────────────────────────────────
print("Loading tokenizer and model...")

model_id = "myshell-ai/Centaur-7B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print("❌ Error loading model/tokenizer:", str(e))
    raise e

# ─────────────────────────────────────
# Handler function
# ─────────────────────────────────────
@validate
def handler(event):
    try:
        print(f"📩 Received event: {event}")

        prompt = event["input"].get("prompt")
        if not prompt:
            return {"error": "Missing 'prompt' in input"}

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("✍️  Input tokenized and moved to device.")

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        print("🧠 Generation complete.")

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"output": result}

    except Exception as e:
        print("❌ Error during inference:", str(e))
        return {"error": str(e)}

# ─────────────────────────────────────
# Register the handler with RunPod
# ─────────────────────────────────────
runpod_handler(handler)