from abc import ABC, abstractmethod

from algorithms.gptfuzzer.fuzzer import PromptNode


class Mutator(ABC):

    def __init__(self, energy: int = 1):
        self.n = energy

    @abstractmethod
    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]) -> 'list[str]':
        pass

    def mutate_batch(self, seeds: list[str], prompt_nodes: list[PromptNode]) -> 'list[list[str]]':
        return [self.mutate_single(seed, prompt_nodes) for seed in seeds]

    @property
    def energy(self):
        return self.n

    @energy.setter
    def energy(self, energy: int):
        self.n = energy


class MutatePolicy(ABC):

    def __init__(self, mutators: 'list[Mutator]'):
        self.mutators = mutators

    @abstractmethod
    def mutate_single(self, prompt_node_seed: PromptNode, prompt_nodes: list[PromptNode]):
        pass

    @abstractmethod
    def mutate_batch(self, prompt_node_seeds: list[PromptNode], prompt_nodes: list[PromptNode]):
        pass