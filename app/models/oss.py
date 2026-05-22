import os

from openai import OpenAI

from app.models.memory import MemoryManager


class OSSAssistant:
    def __init__(self, model_name: str = "kai/deepseek/deepseek-v4-flash:free"):
        """
        Initializes the Open Source Assistant using the FreeTheAI API gateway.
        This bypasses Hugging Face DNS issues while still testing an OSS model.
        """
        api_key = os.getenv("FREETHEAI_API_KEY")
        if not api_key:
            raise ValueError("FREETHEAI_API_KEY environment variable not set.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.freetheai.xyz/v1"
        )
        self.model_name = model_name
        self.memory = MemoryManager()

    def generate_response(self, prompt: str) -> str:
        """Generates a response using the OSS model with memory."""
        self.memory.add_message("user", prompt)
        messages = self.memory.get_context()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            assistant_reply = response.choices[0].message.content
            self.memory.add_message("assistant", assistant_reply)
            return assistant_reply

        except Exception as e:
             return f"Error connecting to OSS API: {str(e)}"
