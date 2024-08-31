import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model
base_model_name = "gpt2"  # Replace with your model's name
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the LoRA configuration and weights
lora_weights_path = "path/to/lora/weights"  # Replace with the path to your LoRA weights
lora_config = PeftConfig.from_pretrained(lora_weights_path)

# Apply the LoRA adapter to the model
model = PeftModel.from_pretrained(model, lora_config)

# Move the model to the desired device (e.g., GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set the model to evaluation mode
model.eval()

# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)