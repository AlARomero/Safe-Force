from fuzzer.Mutators.local_mutators import LocalMutatorCrossOver, LocalMutatorExpand, \
    LocalMutatorGenerateSimilar, LocalMutatorRephrase, LocalMutatorShorten
from fuzzer.Mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from fuzzer.Mutators.policies import MutateRandomSinglePolicy
from fuzzer.core import PromptNode
from utils.llm_utils import get_llm


def build_fuzzer_mutate_policy(model_type: str, model, temperature: float, energy: int) -> MutateRandomSinglePolicy:
    if model_type == "openai":
        return MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(model, temperature, energy),
            OpenAIMutatorExpand(model, temperature, energy),
            OpenAIMutatorGenerateSimilar(model, temperature, energy),
            OpenAIMutatorRephrase(model, temperature, energy),
            OpenAIMutatorShorten(model, temperature, energy)
        ], concatentate=True)
    elif model_type in ("local", "ollama"):
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
    def __init__(self, model_path: str = "llama3:8b", temperature: float = 0.4, energy: int = 1):
        model_type: str = "openai" if "gpt" in model_path else "ollama"
        self.model = get_llm(model_path)
        self.mutate_policy = build_fuzzer_mutate_policy(model_type, self.model, temperature, energy)
        self.temperature = temperature
        self.model_type = model_type
        self.energy = energy

    def change_temperature(self, temperature: float = 0.4):
        self.mutate_policy = build_fuzzer_mutate_policy(self.model_type, self.model, temperature, self.energy)
        self.temperature = temperature

    def change_model(self, model_path: str = "llama3:8b"):
        model_type: str = "openai" if "gpt" in model_path else "ollama"
        self.model = get_llm(model_path)
        self.mutate_policy = build_fuzzer_mutate_policy(model_type, self.model, self.temperature, self.energy)
        self.model_type = model_type

    def mutate(self, seed: PromptNode, from_prompts: list[PromptNode]):
        mutated_prompts: list[PromptNode] = self.mutate_policy.mutate_single(seed, from_prompts)
        return mutated_prompts