from typing import TypedDict

import config
from algorithms.gptfuzzer.fuzzer import GPTFuzzer
from algorithms.gptfuzzer.utils.predict import RoBERTaPredictor
from graph.exploit_result import ExploitResult
from graph.state import GraphState
from prompts.prompt_inject_source import PromptInjectSource
from prompts.utils.gptfuzzer_utils import build_fuzzer_mutate_policy, build_model, build_fuzzer_selection_policy
from prompts.utils.prompt_utils import load_prompts
from algorithms.prompt_inject.promptinject import run_prompts_api, score_attacks


def generate_node(state: GraphState):
    if state.fuzzer:
        raise NotImplementedError("No implementado")
    raise NotImplementedError("No implementado")


def fuzzing_node(state: GraphState):
    raise NotImplementedError("No implementado")


def prompt_inject_node(state: GraphState):
    prompt_inject_config = config.PROMPT_INJECT_CONFIG
    #todo Meter mas api's
    api_model = "openai" if state.target_model.__contains__("gpt") else "ollama"
    prompt_inject_prompts: list[TypedDict] = []
    for prompt in state.prompts:
        if (type(prompt) == PromptInjectSource
        and prompt.settings.attack_text.label == prompt_inject_config.get("label", None)
        and prompt.settings.attack_rogue_string == prompt_inject_config.get("rogue_string", None)
        and prompt.settings.attack_settings_escape == prompt_inject_config.get("escape", None)
        and prompt.settings.attack_settings_delimiter == prompt_inject_config.get("delimiter", None)):
            prompt.settings.base_text.config.model = state.target_model
            prompt.settings.config_model = state.target_model
            prompt.settings.attack_scoring = prompt_inject_config.get("scoring", prompt.settings.attack_scoring)
            dict_prompt = prompt.model_dump()
            prompt_inject_prompts.append(dict_prompt)
    run_prompts_api(prompt_inject_prompts, api_model=api_model)
    score_attacks(prompt_inject_prompts)
    resultados = []
    for prompt in prompt_inject_prompts:
        success: bool = prompt.get("score", 0.0) >= 100
        method: str = f"PromptInject | {prompt.get('settings').get('attack_instruction', 'None')}"
        exploit_prompt: str = prompt.get("settings").get("attack_text", {}).get("instruction", "Getting Attack Text Error")
        text: str = prompt.get("prompt", "Getting Prompt Error")
        resultados.append(ExploitResult(prompt=text, exploit_prompt=exploit_prompt, target_model=state.target_model, method=method, success=success))
    return {"results": resultados}


def gptfuzzer_node(state: GraphState):
    if state.fuzzer:
        gptfuzzer_config = config.GPTFUZZER_CONFIG
        questions = gptfuzzer_config.get("questions", [])
        built_target_model = build_model("ollama" if not state.target_model.startswith("gpt") else "openai", state.target_model)
        fuzzing_model = build_model(gptfuzzer_config["model_type"], gptfuzzer_config["model_path"])
        mutate_policy = build_fuzzer_mutate_policy(gptfuzzer_config["model_type"], fuzzing_model, gptfuzzer_config["temperature"])
        selection_policy = build_fuzzer_selection_policy(gptfuzzer_config["select_policy"])
        seeds: list[str] = [result.prompt for result in state.results if result.success]
        if len(seeds) > 0:
            gptfuzzer = GPTFuzzer(
                questions=questions,
                target=built_target_model,
                predictor=RoBERTaPredictor(gptfuzzer_config["predictor_model"], config.DEVICE),
                initial_seed=seeds,
                mutate_policy=mutate_policy,
                select_policy=selection_policy,
                energy=gptfuzzer_config["energy"],
                max_query=gptfuzzer_config["max_query"],
                max_jailbreak=gptfuzzer_config["max_jailbreak"],
                generate_in_batch=gptfuzzer_config["generate_in_batch"],
            )

            #todo obtener resultado a traves de la funcion run
            results = gptfuzzer.run()
            resultados = []
            if results is not None:
                for result in results:
                    success: bool = result.get("success", False)
                    method: str = "GPTFuzzer"
                    exploit_prompt: str = result["exploit_prompt"]
                    text: str = result["prompt"]
                    resultados.append(ExploitResult(prompt=text, exploit_prompt=exploit_prompt, target_model=state.target_model, method=method, success=success))

            return {"results": resultados}
    return {}

def build_graph():
    raise NotImplementedError("No implementado")


if __name__ == '__main__':
    target_model = "gemma3:1b"
    prompts = load_prompts("prompt_inject")
    fuzzer = True
    initial_state = GraphState(prompts, target_model, fuzzer, results = [])

    prompt_inject_node(initial_state)
    #gptfuzzer_node(initial_state)
    #graph = build_graph()