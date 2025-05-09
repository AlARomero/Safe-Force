import logging
from typing import TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from algorithms.gptfuzzer.fuzzer.Mutators.imutator import Mutator
    from .selection import SelectPolicy
from algorithms.gptfuzzer.llm import LLM
from algorithms.gptfuzzer.utils.template import synthesis_message

class PromptNode:
    def __init__(self,
                 prompt: str,
                 response: Union[list[Tuple[str, str]], list[Tuple[str, list[str]]]] = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.prompt: str = prompt
        self.response: Union[list[Tuple[str, str]], list[Tuple[str, list[str]]]] = response
        self.results: 'list[int]' = results
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1
        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return len(self.results)

"""
def is_stop(self):
    checks = [
        ('max_query', 'current_query'),
        ('max_jailbreak', 'current_jailbreak'),
        ('max_reject', 'current_reject'),
        ('max_iteration', 'current_iteration'),
    ]
    return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)
"""

"""
def run(self):
    logging.info("Fuzzing started!")
    all_generated = []
    try:
        while not self.is_stop():
            seed = self.select_policy.select()
            mutated_results = self.mutate_policy.mutate_single(seed)
            self.evaluate(mutated_results)
            generated_prompts, generated_responses = self.update(mutated_results)
            for p, r in zip(generated_prompts, generated_responses):
                all_generated.append({
                    "prompt": p,
                    "response": r,
                    "exploit_prompt": p,
                    "success": any("jailbreak" in x.lower() for x in r)
                })
            # self.update(mutated_results)
            self.log()
    except KeyboardInterrupt:
        logging.info("Fuzzing interrupted by user")
    logging.info("Fuzzing finished")
    self.raw_fp.close()
    return all_generated
"""

def evaluate(target: LLM, questions: list[str], prompt_nodes: 'list[PromptNode]'):
    for prompt_node in prompt_nodes:
        responses = []
        for question in questions:
            message = synthesis_message(question, prompt_node.prompt)
            if message is None:
                prompt_node.response = []
                prompt_node.results = []
                break
            response = target.generate(message)
            responses.append(response[0] if isinstance(
                response, list) else response)

        prompt_node.response = responses

def update(prompt_nodes: 'list[PromptNode]', select_policy: SelectPolicy):
    select_policy.update(prompt_nodes)

def log(iteration):
    logging.info(
        f"Iteration {iteration}")
