import pandas as pd

from algorithms.gptfuzzer.fuzzer.Mutators.imutator import MutatePolicy
from algorithms.gptfuzzer.fuzzer.Mutators.local_mutators import LocalMutatorCrossOver, LocalMutatorExpand, \
    LocalMutatorGenerateSimilar, LocalMutatorShorten, LocalMutatorRephrase
from algorithms.gptfuzzer.fuzzer.Mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from algorithms.gptfuzzer.fuzzer.Mutators.policies import MutateRandomSinglePolicy
from algorithms.gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy, RoundRobinSelectPolicy, RandomSelectPolicy
from algorithms.gptfuzzer.llm import OpenAILLM, LocalLLM, OllamaLLM, LLM
from prompts.gptfuzzer_source import GPTFuzzConfig
from typing import List
from pydantic import TypeAdapter

def load_prompts_gptfuzzer() -> List[GPTFuzzConfig]:
    with open("./prompts/json/gptfuzzer_prompts.json", "r", encoding="utf-8") as f:
        json_data = f.read()
        adapter = TypeAdapter(list[GPTFuzzConfig])
        prompts = adapter.validate_json(json_data)
    return prompts

def build_mutate_policy(model_type: str, model, temperature: float) -> MutateRandomSinglePolicy:
    if model_type == "openai":
        return MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(model, temperature),
            OpenAIMutatorExpand(model, temperature),
            OpenAIMutatorGenerateSimilar(model, temperature),
            OpenAIMutatorRephrase(model, temperature),
            OpenAIMutatorShorten(model, temperature)
        ], concatentate=True)
    elif model_type in ("local", "ollama"):
        return MutateRandomSinglePolicy([
            LocalMutatorCrossOver(model, temperature),
            LocalMutatorExpand(model, temperature),
            LocalMutatorGenerateSimilar(model, temperature),
            LocalMutatorRephrase(model, temperature),
            LocalMutatorShorten(model, temperature)
        ], concatentate=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def build_model(model_type: str, model_path: str, openai_key: str = None, device: str = "cuda:0") -> LLM:
    if model_type == "openai":
        return OpenAILLM(model_path, openai_key)
    elif model_type == "local":
        return LocalLLM(model_path, device)
    elif model_type == "ollama":
        return LocalLLM(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")