from mutators.imutator import MutatePolicy, Mutator
from graph.prompt_node import PromptNode
import random

class MutateRandomSinglePolicy(MutatePolicy):
    def __init__(self,
                 mutators: 'list[Mutator]',
                 concatentate: bool = True):
        super().__init__(mutators)
        self.concatentate = concatentate

    def mutate_single(self, prompt_node_seed: PromptNode, prompt_nodes: list['PromptNode']) -> 'list[PromptNode]':
        mutator = random.choice(self.mutators)
        results = mutator.mutate_single(prompt_node_seed.prompt, prompt_nodes)
        if self.concatentate:
            results = [result + prompt_node_seed.prompt  for result in results]

        children = [PromptNode(result, parent=prompt_node_seed, mutator=mutator) for result in results]
        prompt_node_seed.child.extend(children)
        return children

    def mutate_batch(self, seeds: list[PromptNode], prompt_nodes):
        raise NotImplementedError("No Implementado")