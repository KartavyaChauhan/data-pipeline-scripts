import gradio as gr
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the recommended model and tokenizer
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model weights to CPU...")
model = AutoModelForCausalLM.from_pretrained(
	model_id,
	torch_dtype=torch.float32, # Ensure CPU compatibility
	device_map="cpu"
)
print("Model loaded successfully!")

# 2. Define the response generation with memory
def generate_response(message, history):
	start_time = time.time()
	
	# Gradio's history is a list of [user_msg, assistant_msg]. We format this into Qwen's expected chat template.
	messages = []
	for user_msg, bot_msg in history:
		messages.append({"role": "user", "content": user_msg})
		messages.append({"role": "assistant", "content": bot_msg})
	
	# Add the current user message
	messages.append({"role": "user", "content": message})

	# Tokenize and format
	text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer([text], return_tensors="pt")

	# Generate response
	generated_ids = model.generate(
		**inputs,
		max_new_tokens=256,
		temperature=0.7,
		do_sample=True
	)
	
	# Extract only the newly generated tokens
	generated_ids = [
		output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
	]

	response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
	
	# Calculate latency for our later report
	latency = round(time.time() - start_time, 2)
	return response + f"\n\n*(Latency: {latency}s)*"

# 3. Create the Public Interface
demo = gr.ChatInterface(
	fn=generate_response,
	title="Ollive AI Bonus: Public OSS Deployment",
	description="Running `Qwen2.5-0.5B-Instruct` natively on a free Hugging Face CPU space. Includes multi-turn memory.",
	theme="soft"
)

if __name__ == "__main__":
	demo.launch()
