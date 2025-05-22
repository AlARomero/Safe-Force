import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
DEVICE: str = os.getenv("DEVICE")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")

TARGETS: list[str] = os.getenv("TARGETS").split("|") if os.getenv("TARGETS") else []
