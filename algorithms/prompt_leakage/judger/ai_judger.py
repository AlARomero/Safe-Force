from algorithms.prompt_leakage.utils import call_llm_general, load_model, parse_llm_response
from algorithms.prompt_leakage.judger.ijudger import Judger

class AIJudger(Judger):
    def __init__(self, model: str, judgement_prompt: str):
        self.model_path: str = model
        self.model = load_model(model)
        self.judgement_prompt = judgement_prompt
        raise NotImplementedError("No implementado")

    def judge(self, case: dict, doc1: str = None, doc2: str = None) -> dict:
        new_case = case.copy()

        #TODO revisar casos de input
        input_text = case['usr_inp - attack vector']

        test_response1 = case['llm_response turn 1']

        llm_input1 = self.judgement_prompt.replace("<INPUT>", input_text).replace("<RESPONSE>", test_response1)

        messages = [
            {"role": "user", "content": llm_input1},
        ]

        test_response1 = call_llm_general(self.model, messages)
        veredict1 = parse_llm_response(test_response1)

        test_response2 = case['llm_response turn 2']

        llm_input2 = self.judgement_prompt.replace("<INPUT>", input_text).replace("<RESPONSE>", test_response2)
        messages = [
            {"role": "user", "content": llm_input2},
        ]

        test_response2 = call_llm_general(self.model, messages)
        veredict2 = parse_llm_response(test_response2)

        #TODO debe devolver label de IA
        case["first_success"] = veredict1
        case["second_success"] = veredict2
        new_case["first_label_success"] = "NO_IA_LABEL"
        new_case["second_label_success"] = "NO_IA_LABEL"

        return new_case