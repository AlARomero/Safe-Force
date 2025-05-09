import csv
import random

from langgraph.graph import StateGraph

import config
from algorithms.gptfuzzer.fuzzer.core import PromptNode
from algorithms.gptfuzzer.fuzzer.selection import RoundRobinSelectPolicy, RandomSelectPolicy, UCBSelectPolicy, \
    MCTSExploreSelectPolicy, EXP3SelectPolicy, SelectPolicy
from algorithms.gptfuzzer.graph.graph_state import GraphState
from algorithms.gptfuzzer.llm import OllamaLLM, OpenAILLM


def generator_node(state: GraphState) -> GraphState:
    mutated_prompts: list[PromptNode] = state.mutator_agent.mutate(state.active_seed, state.actual_list_seeds)
    state.results_generated = mutated_prompts
    return state

def evaluator_node(state: GraphState) -> GraphState:
    state.evaluator_agent.evaluate(state.results_generated, state.questions)
    return state

def classifier_node(state: GraphState) -> GraphState:
    # TODO needs prompts: list[PromptNode], predictor_agent: PredictorAgent
    for prompt in prompts:
        predictor_agent.predict(prompt)

def get_seed_prompt_node(state: GraphState) -> GraphState:
    state.active_seed = state.politica_seleccion.select(state.actual_list_seeds)
    # TODO QEREMOS ELIMINAR SEMILLAS YA UTILIZADAS DEL ARRAY DE SEMILLAS? state.actual_list_seeds.remove(actual_seed)
    return state

def selector_node(state: GraphState) -> GraphState:
    # TODO needs prompts: list[PromptNode]
    selected: list[PromptNode] = []
    for prompt in prompts:
        casos_positivos = len([result for result in prompt.results if result == 1])
        threshold = int(len(prompt.results) * 0.3) # TODO THRESHOLD MODIFICABLE PERO HARDCODEADO
        if casos_positivos >= threshold:
            selected.append(prompt)
    return selected

def strategist_node(state: GraphState) -> GraphState:
    nueva_politica = state.strategist_agent.new_selection_strategy(state.results_generated)
    state.politica_seleccion = nueva_politica

    return state

def logger(state: GraphState) -> GraphState:
    raw_fp = open(state.result_file, 'w', buffering=1, encoding='utf-8')
    writter = csv.writer(raw_fp)
    #writter.writerow(
    #    ['index', 'prompt', 'response', 'parent', 'results'])
    for prompt_node in state.results_generated:
        writter.writerow([prompt_node.index, prompt_node.prompt,
                            prompt_node.response, prompt_node.parent.index, prompt_node.results])
    return state

if __name__ == "__main__":
    target_model = "gemma3:1b"
    energy = 10

    # Construye el grafo
    workflow = StateGraph(state)

    workflow.add_node("Agent Generator", agent_generator)
    workflow.add_node("Evaluator", evaluator)
    workflow.add_node("Agent Predictor", agent_predictor)
    workflow.add_node("Logger", logger)
    workflow.add_node("Selector", selector)
    workflow.add_node("Strategist", strategist)

    workflow.set_entry_point("Agent Generator")

    workflow.add_edge("Agent Generator", "Evaluator")
    workflow.add_edge("Evaluator", "Agent Predictor")
    workflow.add_edge("Agent Predictor", "Logger")
    workflow.add_edge("Agent Predictor", "Selector")
    workflow.add_edge("Selector", "Strategist")

    workflow.add_conditional_edges(
        "Strategist",
        lambda state: "Agent Generator" if state["should_continue"] else END,
    )