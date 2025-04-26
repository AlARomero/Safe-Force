from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

class AttackTool(Enum):
    PROMPT_INJECT = "PromptInject"
    AUTODAN = "AutoDAN"
    LLM_ATTACKS = "LLM-Attacks"

@dataclass
class Prompt:
    """
    Clase base para prompts de ataque para que sea compatible con distintos tipos de ataques
    """
    text: str  # Texto principal del prompt
    tool: AttackTool  # Herramientas de ataque de origen
    metadata: Dict[str, Any] = field(default_factory=dict)  # Datos específicos que maneje la herramienta

    # Metodo para convertir a dict
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tool": self.tool.value,
            "metadata": self.metadata,
        }