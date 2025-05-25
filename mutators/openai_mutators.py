from typing_extensions import Union
import logging

from graph.prompt_node import PromptNode
from mutators.imutator import Mutator
from utils.execution_logger import get_basic_logger
from utils.template_functions import shorten, expand, rephrase, cross_over, generate_similar
from llm import OpenAILLM, AzureOpenAILLM, LocalLLM, OllamaLLM


class OpenAIMutatorBase(Mutator):
    def __init__(self,
                 model: Union[OpenAILLM, AzureOpenAILLM],
                 fallback_mutator_model: Union[LocalLLM, OllamaLLM],
                 logger_name: str = None,
                 temperature: float = 1,
                 max_tokens: int = 512,
                 max_trials: int = 2,
                 energy: int = 1,
                 failure_sleep_time: int = 5):
        super().__init__(energy)

        self.model = model
        self.logger = get_basic_logger(logger_name if logger_name is not None else self.__class__.__name__)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.failure_sleep_time = failure_sleep_time
        self.fallback_mutator_model: Union[LocalLLM, OllamaLLM] = fallback_mutator_model

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]) -> 'list[str]':
        self.logger.info("Mutando")

        response = self.model.generate(seed, self.temperature, self.max_tokens, self.n, self.max_trials, self.failure_sleep_time)

        response = [x for x in response if x is not None and x != " "]
        self.logger.info(f"{len(response)}/{self.n} templates generados con mutacion correctamente")
        diff = self.n - len(response)
        if diff > 0:
            self.logger.warning(f"Han fallado {diff} peticiones de mutacion, mutando con el modelo fallback")
            response.extend(self.fallback_mutator_model.generate(seed, self.temperature, self.max_tokens, n = diff))


        return response


class OpenAIMutatorGenerateSimilar(OpenAIMutatorBase):
    def __init__(self,
                 model: Union[OpenAILLM, AzureOpenAILLM],
                 fallback_model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 max_tokens: int = 512,
                 max_trials: int = 2,
                 energy: int = 1,
                 failure_sleep_time: int = 5):
        super().__init__(model, fallback_model, self.__class__.__name__, temperature, max_tokens, max_trials, energy,  failure_sleep_time)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return super().mutate_single(generate_similar(seed, prompt_nodes), prompt_nodes)


class OpenAIMutatorCrossOver(OpenAIMutatorBase):
    def __init__(self,
                 model: Union[OpenAILLM, AzureOpenAILLM],
                 fallback_model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 max_tokens: int = 512,
                 max_trials: int = 2,
                 energy: int = 1,
                 failure_sleep_time: int = 5):
        super().__init__(model, fallback_model, self.__class__.__name__, temperature, max_tokens, max_trials, energy,  failure_sleep_time)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return super().mutate_single(cross_over(seed, prompt_nodes), prompt_nodes)


class OpenAIMutatorExpand(OpenAIMutatorBase):
    def __init__(self,
                 model: Union[OpenAILLM, AzureOpenAILLM],
                 fallback_model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 max_tokens: int = 512,
                 max_trials: int = 2,
                 energy: int = 1,
                 failure_sleep_time: int = 5):
        super().__init__(model, fallback_model, self.__class__.__name__, temperature, max_tokens, max_trials, energy,  failure_sleep_time)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return [r + seed for r in super().mutate_single(expand(seed, prompt_nodes), prompt_nodes)]


class OpenAIMutatorShorten(OpenAIMutatorBase):
    def __init__(self,
                 model: Union[OpenAILLM, AzureOpenAILLM],
                 fallback_model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 max_tokens: int = 512,
                 max_trials: int = 2,
                 energy: int = 1,
                 failure_sleep_time: int = 5):
        super().__init__(model, fallback_model, self.__class__.__name__, temperature, max_tokens, max_trials, energy,  failure_sleep_time)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return super().mutate_single(shorten(seed, prompt_nodes), prompt_nodes)


class OpenAIMutatorRephrase(OpenAIMutatorBase):
    def __init__(self,
                 model: Union[OpenAILLM, AzureOpenAILLM],
                 fallback_model: Union[LocalLLM, OllamaLLM],
                 temperature: float = 1,
                 max_tokens: int = 512,
                 max_trials: int = 2,
                 energy: int = 1,
                 failure_sleep_time: int = 5):
        super().__init__(model, fallback_model, self.__class__.__name__, temperature, max_tokens, max_trials, energy, failure_sleep_time)

    def mutate_single(self, seed: str, prompt_nodes: list[PromptNode]):
        return super().mutate_single(rephrase(seed, prompt_nodes), prompt_nodes)