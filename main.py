import logging
from typing import TypedDict
import torch
import config
from algorithms.gptfuzzer.fuzzer import GPTFuzzer
from algorithms.gptfuzzer.fuzzer.Mutators.imutator import MutatePolicy
from algorithms.gptfuzzer.fuzzer.Mutators.openai_mutators import OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from algorithms.gptfuzzer.fuzzer.Mutators.policies import MutateRandomSinglePolicy
from algorithms.gptfuzzer.fuzzer.selection import SelectPolicy, MCTSExploreSelectPolicy, RoundRobinSelectPolicy, \
    RandomSelectPolicy
from algorithms.gptfuzzer.llm import OpenAILLM, LocalLLM, LLM
from algorithms.gptfuzzer.utils.predict import Predictor, RoBERTaPredictor
from graph.exploit_result import ExploitResult
from graph.state import GraphState, Fuzzer
from prompts.prompt_inject_source import PromptInjectSource
from prompts.utils.gptfuzzer_utils import load_prompts_gptfuzzer, build_mutate_policy, build_model
from prompts.utils.prompt_utils import load_prompts
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


def gptfuzzer_node(state: GraphState):
    gptfuzzer_config = config.GPTFUZZER_CONFIG
    prompts_gptfuzzer = load_prompts_gptfuzzer()
    questions = gptfuzzer_config.get("questions", [])
    model = build_model(gptfuzzer_config["model_type"], gptfuzzer_config["model_path"])
    mutate_policy = build_mutate_policy(gptfuzzer_config["model_type"], model, gptfuzzer_config["temperature"])
    if gptfuzzer_config["select_policy"] == "mcts":
        select_policy = MCTSExploreSelectPolicy()
    elif gptfuzzer_config["select_policy"] == "round_robin":
        select_policy = RoundRobinSelectPolicy()
    elif gptfuzzer_config["select_policy"] == "random":
        select_policy = RandomSelectPolicy()
    else:
        raise ValueError(f"Unsupported select policy: {gptfuzzer_config['select_policy']}")

    gptfuzzer = GPTFuzzer(
        questions=questions,
        target=model,
        predictor=RoBERTaPredictor('hubert233/GPTFuzz', config.DEVICE),
        initial_seed=prompts_gptfuzzer,
        mutate_policy=mutate_policy,
        select_policy=select_policy,
        energy=gptfuzzer_config["energy"],
        max_query=gptfuzzer_config["max_query"],
        max_jailbreak=gptfuzzer_config["max_jailbreak"],
        generate_in_batch=gptfuzzer_config["generate_in_batch"],
    )

    results = gptfuzzer.run()
    resultados = []
    for result in results:
        exploit_result = ExploitResult(
            prompt=result["prompt"],
            exploit_prompt=result["exploit_prompt"],
            target_model=state.target_model,
            method="GPTFuzzer",
            success=result.get("success", False)
        )
        resultados.append(exploit_result)
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