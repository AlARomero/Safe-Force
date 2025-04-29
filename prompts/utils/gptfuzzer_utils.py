from typing import Union

from algorithms.gptfuzzer.fuzzer.Mutators.local_mutators import LocalMutatorCrossOver, LocalMutatorExpand, \
    LocalMutatorGenerateSimilar, LocalMutatorShorten, LocalMutatorRephrase
from algorithms.gptfuzzer.fuzzer.Mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from algorithms.gptfuzzer.fuzzer.Mutators.policies import MutateRandomSinglePolicy
from algorithms.gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy, RoundRobinSelectPolicy, RandomSelectPolicy
from algorithms.gptfuzzer.llm import OpenAILLM, LocalLLM, LLM, OllamaLLM

def build_fuzzer_mutate_policy(model_type: str, model, temperature: float) -> MutateRandomSinglePolicy:
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
        return OllamaLLM(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def build_fuzzer_selection_policy(policy: str) -> Union[MCTSExploreSelectPolicy, RoundRobinSelectPolicy, RandomSelectPolicy]:
    if policy == "mcts":
        selection_policy = MCTSExploreSelectPolicy()
    elif policy == "round_robin":
        selection_policy = RoundRobinSelectPolicy()
    elif policy == "random":
        selection_policy = RandomSelectPolicy()
    else:
        raise ValueError(f"Unsupported select policy: {policy}")

    return selection_policy