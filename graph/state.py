from dataclasses import dataclass
from typing import Annotated, Union
from operator import add
from graph.exploit_result import ExploitResult
from prompts.prompt_inject_source import PromptInjectSource

@dataclass
class Fuzzer:
    generate_new: bool
    generator_model: str

@dataclass
class GraphState:
    prompts: list[Union[PromptInjectSource]]
    target_model: str
    fuzzer: Fuzzer
    results: Annotated[list[ExploitResult], add]
