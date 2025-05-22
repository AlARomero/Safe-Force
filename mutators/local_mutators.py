from typing import Union

from graph.prompt_node import PromptNode
from mutators.imutator import Mutator
from utils.template_functions import shorten, expand, rephrase, cross_over, generate_similar
from llm import LocalLLM, OllamaLLM


class LocalMutatorBase(Mutator):
    def __init__(self,
                 model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 energy: int = 1,
                 max_tokens: int = 512):
        super().__init__(energy)

        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]) -> 'list[str]':
        return self.model.generate(seed, self.temperature, self.max_tokens, self.n)

class LocalMutatorGenerateSimilar(LocalMutatorBase):
    def __init__(self,
                 model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 energy: int = 1,
                 max_tokens: int = 512):
        super().__init__(model, temperature, energy, max_tokens)

    def mutate_single(self, seed, prompt_nodes):
        return super().mutate_single(generate_similar(seed, prompt_nodes), prompt_nodes)

class LocalMutatorCrossOver(LocalMutatorBase):
    def __init__(self,
                 model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 energy: int = 1,
                 max_tokens: int = 512):
        super().__init__(model, temperature, energy, max_tokens)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return super().mutate_single(cross_over(seed, prompt_nodes), prompt_nodes)

class LocalMutatorExpand(LocalMutatorBase):
    def __init__(self,
                 model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 energy: int = 1,
                 max_tokens: int = 512):
        super().__init__(model, temperature, energy, max_tokens)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return [r + seed for r in super().mutate_single(expand(seed, prompt_nodes), prompt_nodes)]

class LocalMutatorShorten(LocalMutatorBase):
    def __init__(self,
                 model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 energy: int = 1,
                 max_tokens: int = 512):
        super().__init__(model, temperature, energy, max_tokens)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return super().mutate_single(shorten(seed, prompt_nodes), prompt_nodes)

class LocalMutatorRephrase(LocalMutatorBase):
    def __init__(self,
                 model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 energy: int = 1,
                 max_tokens: int = 512):
        super().__init__(model, temperature, energy, max_tokens)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return super().mutate_single(rephrase(seed, prompt_nodes), prompt_nodes)