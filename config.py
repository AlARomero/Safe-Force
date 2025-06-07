import csv
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
DEVICE: str = os.getenv("DEVICE")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
TARGETS: list[str] = os.getenv("TARGETS").split("|") if os.getenv("TARGETS") else []
LOG_FILE_PATH = f"logs/log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULT_FILE_PATH = f"results/results_{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}.csv"
STRATEGIST_TEMPERATURE: float = float(os.getenv("STRATEGIST_TEMPERATURE"))
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS"))
ENERGY: int = int(os.getenv("ENERGY"))
POSSIBLE_MUTATORS: list[str] = os.getenv("POSSIBLE_MUTATORS").split("|") if os.getenv("POSSIBLE_MUTATORS") else []
QUESTIONS_USED_NUMBER: int = int(os.getenv("QUESTIONS_USED_NUMBER"))
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE")
PROMPT_TEMPLATES_FILE = os.getenv("PROMPT_TEMPLATES_FILE")
PROMPT_TEMPLATES: list[str] = []
QUESTIONS: list[str] = []

if PROMPT_TEMPLATES_FILE:
    with open(PROMPT_TEMPLATES_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        PROMPT_TEMPLATES = [row["text"] for row in reader if "text" in row and row["text"].strip()]

if QUESTIONS_FILE:
    with open(QUESTIONS_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        QUESTIONS = [row["text"] for row in reader if "text" in row and row["text"].strip()]