from typing import Optional, List, Any
from pydantic import BaseModel

class Config(BaseModel):
    temperature: float
    top_p: float
    max_tokens: int
    presence_penalty: float
    model: str
    frequency_penalty: float

class BaseText(BaseModel):
    instruction: str
    label: str
    input: str
    config: Config
    type: str

class AttackText(BaseModel):
    label: str
    instruction: str

class Settings(BaseModel):
    base_text: BaseText
    prompt_secret: Optional[str]
    prompt_secret_instruction: Optional[str]
    prompt_private_value: Optional[str]
    prompt_instruction: str
    prompt_shots: List[Any]  # Cambiar si sabes qué estructura tiene
    prompt_input: str
    prompt_nshot: Optional[int]
    prompt_label: str
    prompt_terms_human: str
    prompt_terms_ai: str
    attack_text: AttackText
    attack_instruction: str
    attack_scoring: str
    attack_rogue_string: str
    attack_label: str
    attack_settings_escape: str
    attack_settings_delimiter: str
    attack_settings_escape_length: int
    attack_settings_escape_times: int
    config_model: str
    config_temperature: float
    config_top_p: float
    config_frequency_penalty: float
    config_presence_penalty: float
    config_max_tokens: int
    config_stop: Optional[list[str]]
    config_id: int
    vis_columns: List[str]

class PromptInjectSource(BaseModel):
    hash: str
    settings: Settings
    prompt: str