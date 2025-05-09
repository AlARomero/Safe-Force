import csv
import random

from langgraph.graph import StateGraph

import config
from algorithms.gptfuzzer.agents.evaluator_agent import EvaluatorAgent
from algorithms.gptfuzzer.agents.mutator_agent import MutatorAgent
from algorithms.gptfuzzer.agents.predictor_agent import PredictorAgent
from algorithms.gptfuzzer.agents.strategist_agent import StrategistAgent
from algorithms.gptfuzzer.fuzzer.core import PromptNode
from algorithms.gptfuzzer.fuzzer.selection import RoundRobinSelectPolicy, SelectPolicy
from algorithms.gptfuzzer.graph.graph_state import GraphState


def generator_node(state: GraphState) -> GraphState:
    mutated_prompts: list[PromptNode] = state.mutator_agent.mutate(state.active_seed, state.actual_list_seeds)
    state.results_generated = mutated_prompts
    return state

def evaluator_node(state: GraphState) -> GraphState:
    state.evaluator_agent.evaluate(state.results_generated, state.questions)
    return state

def classifier_node(state: GraphState) -> GraphState:
    for prompt in state.results_generated:
        prompt.results = []
        for model, responses in prompt.response:
            prompt.results.append(state.predictor_agent.predict(responses if isinstance(responses, list) else [responses]))
    return state

def get_seed_prompt_node(state: GraphState) -> GraphState:
    state.active_seed = state.politica_seleccion.select(state.actual_list_seeds) # Coge semilla de la lista actual
    # TODO QEREMOS ELIMINAR SEMILLAS YA UTILIZADAS DEL ARRAY DE SEMILLAS? state.actual_list_seeds.remove(actual_seed)
    return state

def selector_node(state: GraphState) -> GraphState:
    # TODO needs prompts: list[PromptNode]
    selected: list[PromptNode] = []
    for prompt in state.results_generated:
        casos_positivos = len([result for model, results in prompt.results for result in results  if result == 1]) # TODO Supongo que se podra mejorar el algoritmo
        threshold = int(len(prompt.results) * 0.3) # TODO THRESHOLD MODIFICABLE PERO HARDCODEADO
        if casos_positivos >= threshold:
            selected.append(prompt)
    state.actual_list_seeds = selected # Añade las seleccionadas a la lista actual
    return state

def strategist_node(state: GraphState) -> GraphState:
    nueva_politica = state.strategist_agent.new_selection_strategy(state.results_generated)
    state.politica_seleccion = nueva_politica

    return state

def logger_node(state: GraphState) -> GraphState:
    raw_fp = open(state.result_file, 'w', buffering=1, encoding='utf-8')
    writter = csv.writer(raw_fp)
    #writter.writerow(
    #    ['index', 'prompt', 'response', 'parent', 'results'])
    for prompt_node in state.results_generated:
        writter.writerow([prompt_node.index, prompt_node.prompt,
                            prompt_node.response, prompt_node.parent.index, prompt_node.results])
    return state

if __name__ == "__main__":

    targets: list[str] = ["llama3:8b", "gemma3:1b"]
    strategist_model_path: str = "llama3:8b"
    predictor_model_path: str = "hubert233/GPTFuzz"
    mutator_model_path: str = "llama3:8b"
    mutator_temperature: float = 0.4
    politica_seleccion: SelectPolicy = RoundRobinSelectPolicy()

    initial_state = GraphState(
        initial_seeds = [],
        actual_list_seeds = [],
        active_seed = None,
        energy = 10,
        politica_seleccion = politica_seleccion,
        results_generated = [],
        result_file = "resultados.csv",
        generated = 0,
        strategist_agent = StrategistAgent(politica_seleccion, strategist_model_path),
        predictor_agent = PredictorAgent(predictor_model_path, config.DEVICE),
        mutator_agent = MutatorAgent(mutator_model_path, mutator_temperature),
        evaluator_agent = EvaluatorAgent(targets),
        questions = []
    )

    # Construye el grafo
    builder = StateGraph(GraphState)

    builder.add_node("Generator", generator_node)
    builder.add_node("Evaluator", evaluator_node)
    builder.add_node("Classifier", classifier_node)
    builder.add_node("Logger", logger_node)
    builder.add_node("Selector", selector_node)
    builder.add_node("Strategist", strategist_node)

    builder.set_entry_point("Agent Generator")

    builder.add_edge("Agent Generator", "Evaluator")
    builder.add_edge("Evaluator", "Agent Predictor")
    builder.add_edge("Agent Predictor", "Logger")
    builder.add_edge("Agent Predictor", "Selector")
    builder.add_edge("Selector", "Strategist")

    builder.add_conditional_edges(
        "Strategist",
        lambda state: "Agent Generator" if state["should_continue"] else END,
    )

    grafo = builder.compile()

    grafo.invoke(initial_state)