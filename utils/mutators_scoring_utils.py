import random
from argparse import ArgumentError
from typing import Union, Tuple

from mutators import LocalMutatorCrossOver, LocalMutatorExpand, \
    LocalMutatorGenerateSimilar, LocalMutatorRephrase, LocalMutatorShorten
from mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from graph.prompt_node import PromptNode
from llm import LLM, OpenAILLM, AzureOpenAILLM, LocalLLM, OllamaLLM
from mutators.imutator import Mutator

def initialize_mutators_score(possible_mutators: list[str]) -> dict[str, float]:
    scores = {}
    for mutator_type in possible_mutators:
        scores[mutator_type] = 1.0 / len(possible_mutators)
    return scores

def build_mutator(mutator_name: str, model: LLM, temperature: float, energy: int, fallback_model: Union[LocalLLM, OllamaLLM] = None) -> Mutator:
    model_type = type(model)
    if isinstance(model, (OpenAILLM, AzureOpenAILLM)):
        if fallback_model is None:
            raise ArgumentError(None, "Fallback model Null para mutador Openai")
        mapa = {
            "MutatorCrossOver": OpenAIMutatorCrossOver(model, fallback_model, temperature, energy=energy),
            "MutatorExpand": OpenAIMutatorExpand(model, fallback_model, temperature, energy=energy),
            "MutatorGenerateSimilar": OpenAIMutatorGenerateSimilar(model, fallback_model, temperature, energy=energy),
            "MutatorRephrase": OpenAIMutatorRephrase(model, fallback_model, temperature, energy=energy),
            "MutatorShorten": OpenAIMutatorShorten(model, fallback_model, temperature, energy=energy)
        }
    elif isinstance(model, (OllamaLLM, LocalLLM)):
        mapa = {
            "MutatorCrossOver": LocalMutatorCrossOver(model, temperature, energy),
            "MutatorExpand": LocalMutatorExpand(model, temperature, energy),
            "MutatorGenerateSimilar": LocalMutatorGenerateSimilar(model, temperature, energy),
            "MutatorRephrase": LocalMutatorRephrase(model, temperature, energy),
            "MutatorShorten": LocalMutatorShorten(model, temperature, energy)
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return mapa[mutator_name]

def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        # Evita división por cero: distribuye peso uniformemente
        n = len(weights)
        return {k: 1.0 / n for k in weights}

    return {k: v / total for k, v in weights.items()}

def get_mutator_by_weight(seed: PromptNode) -> str:
    suma: float = 0.0
    ranges: list[Tuple[str, float]] = []
    for key, value in seed.mutators_score.items():
        ranges.append((key, value + suma))
        suma += value

    valor: float = random.random()
    for (mutator, range_value) in ranges:
        if range_value > valor:
            return mutator
    return ""