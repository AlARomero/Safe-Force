from typing import TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from fuzzer.Mutators.imutator import Mutator


class PromptNode:
    def __init__(self,
                 prompt: str,
                 response: Union[list[Tuple[str, str]], list[Tuple[str, list[str]]]] = None,
                 results: 'list[Tuple[str, int]]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.prompt: str = prompt
        self.response: Union[list[Tuple[str, str]], list[Tuple[str, list[str]]]] = response
        self.results: 'list[Tuple[str, list[int]]]' = results # [(model, [0, 1, ...]), ...]
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

    @property
    def num_jailbreak(self):
        entero: int = sum([sum(lista_resultados) for model, lista_resultados in self.results])
        return entero

    @property
    def num_query(self):
        total: int = 0
        for model, results in self.results:
            total += len(results)
        return total

    @property
    def num_reject(self):
        return self.num_query - self.num_jailbreak