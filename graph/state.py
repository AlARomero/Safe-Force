from dataclasses import dataclass
from typing import Annotated
from operator import add

from graph.data.exploit_result import ExploitResult
from graph.data.prompt import Prompt


@dataclass
class GraphState:
    prompts: list[Prompt]
    results: Annotated[list[ExploitResult], add]