from typing import List, Dict, Optional, Literal
from pydantic import BaseModel


class GPTFuzzConfig(BaseModel):
    model_type: Literal["openai", "local", "ollama"]
    model_path: str
    openai_key: Optional[str]
    prompts_json_path: str
    temperature: float
    energy: int
    max_query: int
    max_jailbreak: int
    max_reject: int
    max_iteration: int
    generate_in_batch: bool
    device: str
    result_file: Optional[str]
    questions: List[str]