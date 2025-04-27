from typing import TypedDict

import config
from graph.exploit_result import ExploitResult
from graph.state import GraphState, Fuzzer

from langgraph.graph import StateGraph

from prompts.prompt_inject_source import PromptInjectSource
from prompts.prompt_utils import load_prompts

from algorithms.prompt_inject.promptinject import run_prompts_api, score_attacks

def generate_node(state: GraphState):
    if state.fuzzer.generate_new:
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
    for prompt in prompts:
        success: bool = prompt.get("score", 0.0) >= 100
        method: str = f"PromptInject | {prompt.get('settings').get('attack_instruction', 'None')}"
        exploit_prompt: str = prompt.get("settings").get("attack_text", {}).get("instruction", "Getting Attack Text Error")
        text: str = prompt.get("prompt", "Getting Prompt Error")
        resultados.append(ExploitResult(prompt=text, exploit_prompt=exploit_prompt, target_model=state.target_model, method=method, success=success))

    return {"results": resultados}


def build_graph():
    raise NotImplementedError("No implementado")

if __name__ == '__main__':
    target_model = "gemma3:1b"
    prompts = load_prompts("prompt_inject")
    prompts = prompts + load_prompts("llm-attack")

    initial_state = GraphState(prompts, target_model, Fuzzer(False, ""), [])

    prompt_inject_node(initial_state)
    #graph = build_graph()