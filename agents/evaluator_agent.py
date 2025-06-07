import random
from typing import Tuple
from graph.prompt_node import PromptNode
from llm.llm import LLM
from utils.llm_utils import get_llm
from utils.logger import get_basic_logger
from utils.template import synthesis_message


class EvaluatorAgent:
    def __init__(self, targets_model_path: list[str], questions: list[str], questions_used_number: int):
        self.targets: list[LLM] = [get_llm(model_path) for model_path in targets_model_path]
        self.logger = get_basic_logger(self.__class__.__name__)
        self.questions = questions
        self.questions_used_number = questions_used_number
    def evaluate(self, prompts: list[PromptNode]):
        for prompt_node in prompts:
            responses: list[Tuple[str, list[str]]] = [(model.model_path, []) for model in self.targets]
            prompt_node.questions = random.sample(self.questions, self.questions_used_number)
            synthetic_questions = [synthesis_message(question, prompt_node.prompt) for question in prompt_node.questions]
            for i, (model_path, response_list) in enumerate(responses):
                target = next(model for model in self.targets if model.model_path == model_path)
                for message in synthetic_questions:
                    if message is None:
                        self.logger.warning(f"Mensaje Nulo. Prompt {prompt_node.index} posiblemente inválido.")
                        prompt_node.response = []
                        prompt_node.results = []
                        break
                    else:
                        response = target.generate(message)
                        response_list.append(response[0] if isinstance(response, list) else response)
                        responses[i] = (model_path, response_list)
            prompt_node.response = responses