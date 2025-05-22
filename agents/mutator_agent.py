from mutators import LocalMutatorCrossOver, LocalMutatorExpand, \
    LocalMutatorGenerateSimilar, LocalMutatorRephrase, LocalMutatorShorten
from mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from mutators import MutateRandomSinglePolicy
from graph.prompt_node import PromptNode
from llm import LLM, OpenAILLM, AzureOpenAILLM, LocalLLM, OllamaLLM
from utils.llm_utils import get_llm


def build_fuzzer_mutate_policy(model: LLM, temperature: float, energy: int) -> MutateRandomSinglePolicy:
    model_type = type(model)
    if isinstance(model, (OpenAILLM, AzureOpenAILLM)):
        return MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(model, temperature, energy),
            OpenAIMutatorExpand(model, temperature, energy),
            OpenAIMutatorGenerateSimilar(model, temperature, energy),
            OpenAIMutatorRephrase(model, temperature, energy),
            OpenAIMutatorShorten(model, temperature, energy)
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
    def __init__(self, model_path: str = "ollama/llama3:8b", temperature: float = 0.4, energy: int = 1):
        self.model = get_llm(model_path)

        self.mutate_policy = build_fuzzer_mutate_policy(self.model, temperature, energy)
        self.temperature = temperature
        self.energy = energy

    def change_temperature(self, temperature: float = 0.4):
        self.mutate_policy = build_fuzzer_mutate_policy(self.model, temperature, self.energy)
        self.temperature = temperature

    def change_model(self, model_path: str = "ollama/llama3:8b"):
        self.model = get_llm(model_path)
        self.mutate_policy = build_fuzzer_mutate_policy(self.model, self.temperature, self.energy)

    def mutate(self, seed: PromptNode, from_prompts: list[PromptNode]):
        mutated_prompts: list[PromptNode] = self.mutate_policy.mutate_single(seed, from_prompts)
        return mutated_prompts