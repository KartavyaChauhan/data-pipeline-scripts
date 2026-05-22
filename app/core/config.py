import os

from dotenv import load_dotenv


# Force load environment variables from .env file.
load_dotenv()

FREETHEAI_API_KEY = os.getenv("FREETHEAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
