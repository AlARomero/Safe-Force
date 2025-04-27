from abc import ABC, abstractmethod
from algorithms.gptfuzzer.fuzzer.core import GPTFuzzer

class Mutator(ABC):

    def __init__(self, fuzzer: 'GPTFuzzer'):
        self._fuzzer = fuzzer
        self.n = None

    @abstractmethod
    def mutate_single(self, seed) -> 'list[str]':
        pass

    def mutate_batch(self, seeds) -> 'list[list[str]]':
        return [self.mutate_single(seed) for seed in seeds]

    @property
    def fuzzer(self):
        return self._fuzzer

    @fuzzer.setter
    def fuzzer(self, gptfuzzer):
        self._fuzzer = gptfuzzer
        self.n = gptfuzzer.energy


class MutatePolicy(ABC):

    def __init__(self, mutators: 'list[Mutator]', fuzzer: 'GPTFuzzer' = None):
        self.mutators = mutators
        self._fuzzer = fuzzer

    @abstractmethod
    def mutate_single(self, seed):
        pass

    @abstractmethod
    def mutate_batch(self, seeds):
        pass

    @property
    def fuzzer(self):
        return self._fuzzer

    @fuzzer.setter
    def fuzzer(self, gptfuzzer):
        self._fuzzer = gptfuzzer
        for mutator in self.mutators:
            mutator.fuzzer = gptfuzzer