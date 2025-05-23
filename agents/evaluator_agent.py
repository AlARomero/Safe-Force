from typing import Tuple

from graph.prompt_node import PromptNode
from llm.llm import LLM
from utils.llm_utils import get_llm
from utils.template import synthesis_message


class EvaluatorAgent:
    def __init__(self, targets_model_path: list[str], questions: list[str]):
        self.targets: list[LLM] = [get_llm(model_path) for model_path in targets_model_path]
        self.questions = questions
    def evaluate(self, prompts: list[PromptNode]):
        for prompt_node in prompts:
            responses: list[Tuple[str, list[str]]] = [(model.model_path, []) for model in self.targets]
            synthetic_questions = [synthesis_message(question, prompt_node.prompt) for question in self.questions]
            for i, (model_path, response_list) in enumerate(responses):
                target = next(model for model in self.targets if model.model_path == model_path)
                for message in synthetic_questions:
                    if message is None:
                        prompt_node.response = []
                        prompt_node.results = []
                        break
                    else:
                        response = target.generate(message)
                        response_list.append(response[0] if isinstance(response, list) else response)
                        responses[i] = (model_path, response_list)
            prompt_node.response = responses
