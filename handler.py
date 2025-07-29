import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.handler import runpod_handler

# ─────────────────────────────────────
# Setup logging
# ─────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────
# Load model at cold start
# ─────────────────────────────────────
logger.info("Loading tokenizer and model...")

model_id = "myshell-ai/Centaur-7B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.exception("Error loading model/tokenizer:")
    raise e

# ─────────────────────────────────────
# Handler function
# ─────────────────────────────────────
@validate
def handler(event):
    try:
        logger.info(f"Received event: {event}")

        prompt = event["input"].get("prompt")
        if not prompt:
            return {"error": "Missing 'prompt' in input"}

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logger.info("Input tokenized and moved to model device.")

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Generation complete.")

        return {"output": result}

    except Exception as e:
        logger.exception("Error during inference:")
        return {"error": str(e)}

# ─────────────────────────────────────
# Register the handler with RunPod
# ─────────────────────────────────────
runpod_handler(handler)