from algorithms.gptfuzzer.fuzzer.Mutators.local_mutators import LocalMutatorCrossOver, LocalMutatorExpand, \
    LocalMutatorGenerateSimilar, LocalMutatorRephrase, LocalMutatorShorten
from algorithms.gptfuzzer.fuzzer.Mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from algorithms.gptfuzzer.fuzzer.Mutators.policies import MutateRandomSinglePolicy
from algorithms.gptfuzzer.fuzzer.core import PromptNode
from algorithms.gptfuzzer.utils.llm_utils import get_llm


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

class MutatorAgent:
    def __init__(self, model_path: str = "llama3:8b", temperature: float = 0.4):
        model_type: str
        if model_path.__contains__("gpt"):
            model_type = "openai"
        else:
            model_type = "ollama"
        self.model = get_llm(model_path)
        self.mutate_policy = build_fuzzer_mutate_policy(model_type, self.model, temperature)
        self.temperature = temperature
        self.model_type = model_type

    def change_temperature(self, temperature: float = 0.4):
        self.mutate_policy = build_fuzzer_mutate_policy(self.model_type, self.model, temperature)
        self.temperature = temperature

    def change_model(self, model_path: str = "llama3:8b"):
        model_type: str
        if model_path.__contains__("gpt"):
            model_type = "openai"
        else:
            model_type = "ollama"
        self.model = get_llm(model_path)
        self.mutate_policy = build_fuzzer_mutate_policy(model_type, self.model, self.temperature)
        self.model_type = model_type

    def mutate(self, seed: PromptNode, from_prompts: list[PromptNode]):
        mutated_prompts: list[PromptNode] = self.mutate_policy.mutate_single(seed, from_prompts)
        return mutated_prompts