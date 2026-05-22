import uuid

import gradio as gr
import requests

BACKEND_URL = "http://localhost:8000/api"

SESSION_ID = str(uuid.uuid4())


def respond(message, chat_history_hosted, chat_history_oss):
	if not message.strip():
		return "", chat_history_hosted, chat_history_oss

	try:
		payload = {"session_id": SESSION_ID, "prompt": message}
		response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)

		if response.status_code == 200:
			data = response.json()
			hosted_reply = data["hosted_response"]
			oss_reply = data["oss_response"]
		else:
			detail = response.json().get("detail", "Unknown error")
			hosted_reply = f"Error: {detail}"
			oss_reply = f"Error: {detail}"
	except Exception as e:
		hosted_reply = f"Failed to connect to backend: {str(e)}"
		oss_reply = f"Failed to connect to backend: {str(e)}"

	chat_history_hosted.append((message, hosted_reply))
	chat_history_oss.append((message, oss_reply))

	return "", chat_history_hosted, chat_history_oss


def clear_chat():
	try:
		requests.post(f"{BACKEND_URL}/clear", json={"session_id": SESSION_ID, "prompt": ""}, timeout=30)
	except Exception:
		pass
	return [], []


with gr.Blocks(title="AI Personal Assistant Benchmarking") as demo:
	gr.Markdown("# 🤖 Ollive AI Assistant Benchmarking Platform")
	gr.Markdown(
		"Test side-by-side performance of **Frontier Models (GPT-4.1 via FreeTheAI)** vs **Open Source Models (Qwen 2.5-72B via HF)**."
	)

	with gr.Row():
		with gr.Column():
			gr.Markdown("### 🚀 Frontier Assistant (bbl/gpt-4.1)")
			hosted_chatbot = gr.Chatbot(label="GPT-4.1 Context Stream")

		with gr.Column():
			gr.Markdown("### 🌐 Open Source Assistant (Qwen 2.5-72B)")
			oss_chatbot = gr.Chatbot(label="Qwen 2.5 Context Stream")

	msg = gr.Textbox(placeholder="Type your message or evaluation prompt here and press Enter...", label="Universal Input Layer")

	with gr.Row():
		submit_btn = gr.Button("Send to Both Assistants", variant="primary")
		clear_btn = gr.Button("Clear Conversational Memory")

	msg.submit(respond, [msg, hosted_chatbot, oss_chatbot], [msg, hosted_chatbot, oss_chatbot])
	submit_btn.click(respond, [msg, hosted_chatbot, oss_chatbot], [msg, hosted_chatbot, oss_chatbot])
	clear_btn.click(clear_chat, None, [hosted_chatbot, oss_chatbot])


if __name__ == "__main__":
	demo.launch(server_name="0.0.0.0", server_port=7860)
