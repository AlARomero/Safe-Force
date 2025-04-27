from algorithms.gptfuzzer.fuzzer.core import GPTFuzzer
from algorithms.gptfuzzer.fuzzer.Mutators.imutator import Mutator
from algorithms.gptfuzzer.utils.template_functions import shorten, expand, rephrase, cross_over, generate_similar
from algorithms.gptfuzzer.llm import OpenAILLM


class OpenAIMutatorBase(Mutator):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(fuzzer)

        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.failure_sleep_time = failure_sleep_time

    def mutate_single(self, seed) -> 'list[str]':
        return self.model.generate(seed, self.temperature, self.max_tokens, self.n, self.max_trials, self.failure_sleep_time)


class OpenAIMutatorGenerateSimilar(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(generate_similar(seed, self.fuzzer.prompt_nodes))


class OpenAIMutatorCrossOver(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(cross_over(seed, self.fuzzer.prompt_nodes))


class OpenAIMutatorExpand(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)

    def mutate_single(self, seed):
        return [r + seed for r in super().mutate_single(expand(seed, self.fuzzer.prompt_nodes))]


class OpenAIMutatorShorten(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(shorten(seed, self.fuzzer.prompt_nodes))


class OpenAIMutatorRephrase(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(rephrase(seed, self.fuzzer.prompt_nodes))