from argparse import ArgumentError
from typing import Union

from mutators import LocalMutatorCrossOver, LocalMutatorExpand, \
    LocalMutatorGenerateSimilar, LocalMutatorRephrase, LocalMutatorShorten
from mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from mutators import MutateRandomSinglePolicy
from graph.prompt_node import PromptNode
from llm import LLM, OpenAILLM, AzureOpenAILLM, LocalLLM, OllamaLLM
from utils.llm_utils import get_llm


def build_fuzzer_mutate_policy(model: LLM, temperature: float, energy: int, fallback_model: Union[LocalLLM, OllamaLLM] = None) -> MutateRandomSinglePolicy:
    model_type = type(model)
    if isinstance(model, (OpenAILLM, AzureOpenAILLM)):
        if fallback_model is None:
            raise ArgumentError(None, "Fallback model Null para mutador Openai")
        return MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(model, fallback_model, temperature, energy=energy),
            OpenAIMutatorExpand(model, fallback_model, temperature, energy=energy),
            OpenAIMutatorGenerateSimilar(model, fallback_model, temperature, energy=energy),
            OpenAIMutatorRephrase(model, fallback_model, temperature, energy=energy),
            OpenAIMutatorShorten(model, fallback_model, temperature, energy=energy)
        ], concatentate=True)
    elif isinstance(model, (OllamaLLM, LocalLLM)):
        return MutateRandomSinglePolicy([
            LocalMutatorCrossOver(model, temperature, energy),
            LocalMutatorExpand(model, temperature, energy),
            LocalMutatorGenerateSimilar(model, temperature, energy),
            LocalMutatorRephrase(model, temperature, energy),
            LocalMutatorShorten(model, temperature, energy)
        ], concatentate=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class MutatorAgent:
    def __init__(self, model_path: str = "ollama/llama3:8b", fallback_model_path: str = None, temperature: float = 0.4, energy: int = 1):
        self.model = get_llm(model_path)
        self.fallback_model = get_llm(fallback_model_path) if fallback_model_path is not None else None
        self.mutate_policy = build_fuzzer_mutate_policy(self.model, temperature, energy, self.fallback_model)
        self.temperature = temperature
        self.energy = energy

    def change_temperature(self, temperature: float = 0.4):
        self.mutate_policy = build_fuzzer_mutate_policy(self.model, temperature, self.energy, self.fallback_model)
        self.temperature = temperature

    def change_model(self, model_path: str = "ollama/llama3:8b"):
        self.model = get_llm(model_path)
        self.mutate_policy = build_fuzzer_mutate_policy(self.model, self.temperature, self.energy, self.fallback_model)

    def mutate(self, seed: PromptNode, from_prompts: list[PromptNode]):
        mutated_prompts: list[PromptNode] = self.mutate_policy.mutate_single(seed, from_prompts)
        return mutated_prompts