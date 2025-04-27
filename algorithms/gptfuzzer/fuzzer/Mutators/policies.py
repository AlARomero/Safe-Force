from algorithms.gptfuzzer.fuzzer.Mutators.imutator import MutatePolicy, Mutator
from algorithms.gptfuzzer.fuzzer.core import GPTFuzzer, PromptNode
import random

class MutateRandomSinglePolicy(MutatePolicy):
    def __init__(self,
                 mutators: 'list[Mutator]',
                 fuzzer: 'GPTFuzzer' = None,
                 concatentate: bool = True):
        super().__init__(mutators, fuzzer)
        self.concatentate = concatentate

    def mutate_single(self, prompt_node: 'PromptNode') -> 'list[PromptNode]':
        mutator = random.choice(self.mutators)
        results = mutator.mutate_single(prompt_node.prompt)
        if self.concatentate:
            results = [result + prompt_node.prompt  for result in results]

        return [PromptNode(self.fuzzer, result, parent=prompt_node, mutator=mutator) for result in results]

    def mutate_batch(self, seeds):
        raise NotImplementedError("No Implementado")