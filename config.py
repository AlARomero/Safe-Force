import os
import datetime
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
DEVICE: str = os.getenv("DEVICE")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")

TARGETS: list[str] = os.getenv("TARGETS").split("|") if os.getenv("TARGETS") else []

LOG_FILE_PATH = f"logs/log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULT_FILE_PATH = f"resultados/resultados_{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}.csv"
