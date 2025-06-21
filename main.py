from typing import Optional
import traceback
import config
from agents.evaluator_agent import EvaluatorAgent
from agents.mutator_agent import MutatorAgent
from agents.predictor_agent import PredictorAgent
from agents.strategist_agent import StrategistAgent
from graph.safe_force_graph import build_graph, write_file
from graph.prompt_node import PromptNode
from llm import LocalLLM
from selection.selection import RoundRobinSelectPolicy, SelectPolicy
from graph.graph_state import GraphState
from utils.logger import get_basic_logger
from utils.mutators_scoring_utils import initialize_mutators_score

def run(targets: list[str]) -> Optional[GraphState]:
    logger = get_basic_logger("Safe-Force-Graph")
    targets: list[str] = targets
    strategist_model_path: str = "azure/gpt-4-TFM"
    strategist_model_endpoint: str = config.AZURE_OPENAI_ENDPOINT
    strategist_model_api_key: str = config.AZURE_OPENAI_API_KEY
    strategist_temperature: float = config.STRATEGIST_TEMPERATURE
    predictor_model_path: str = "hubert233/GPTFuzz"
    mutator_model_path: str = "azure/gpt-4-TFM"
    fallback_mutator_model_path: str = "ollama/llama3:8b"
    mutator_temperature: float = 0.4
    energy = config.ENERGY
    max_iterations: int = config.MAX_ITERATIONS
    mutators_default_score = initialize_mutators_score(config.POSSIBLE_MUTATORS)
    politica_seleccion: SelectPolicy = RoundRobinSelectPolicy()
    questions_used_number = config.QUESTIONS_USED_NUMBER
    all_questions: list[str] = config.QUESTIONS
    write_file(config.RESULT_FILE_PATH, [], politica_seleccion.__class__.__name__, True)
    nodes: list[PromptNode] = [PromptNode(prompt["prompt_text"], prompt["prompt_type"], index=i, mutators_score=mutators_default_score.copy()) for i, prompt in enumerate(config.PROMPT_TEMPLATES)]
    initial_state = GraphState(
        initial_seeds = nodes,
        actual_list_seeds = nodes.copy(),
        active_seed = None,
        energy = energy,
        politica_seleccion = politica_seleccion,
        results_generated = [],
        result_file = config.RESULT_FILE_PATH,
        generated = 0,
        strategist_agent = StrategistAgent(politica_seleccion, questions_used_number, max_iterations, strategist_model_path, strategist_temperature, strategist_model_endpoint, strategist_model_api_key),
        predictor_agent = PredictorAgent(predictor_model_path, config.DEVICE),
        mutator_agent = MutatorAgent(mutators_default_score, mutator_model_path, fallback_mutator_model_path, mutator_temperature, energy),
        evaluator_agent = EvaluatorAgent(targets, all_questions, questions_used_number),
        should_continue = True,
        logger = logger
    )

    logger.info("Compilando grafo.")
    try:
        grafo = build_graph()
        logger.info("Grafo compilado exitosamente.")
    except Exception as e:
        logger.error(f"Error al compilar grafo: {str(e)}")
        raise
    logger.info("Iniciando fuzzing.")
    try:
        result_dict = grafo.invoke(vars(initial_state), {"recursion_limit": 500})
        logger.info("Condiciones de parada alcanzadas. Finalizando ejecución.")
        final_state = GraphState.from_dict(result_dict)
        logger.info("Ejecución completada exitosamente.")
        total_jailbreaks = sum(p.num_jailbreak for p in final_state.results_generated)
        total_generated_prompts = sum(p.num_query for p in final_state.results_generated)
        total_rejections = sum(p.num_reject for p in final_state.results_generated)
        logger.info(f"Resumen final - Iteraciones: {final_state.strategist_agent.iterations} | Jailbreaks totales: {total_jailbreaks}. | Rechazos en esta iteración: {total_rejections}. | Respuestas totales: {total_generated_prompts}.")
        return final_state
    except Exception as e:
        logger.error(f"Error durante ejecución:\n{traceback.format_exc()}\nEstado final en la iteración {initial_state.strategist_agent.iterations}.")
        return None

def test_local_llm_generate_response(device: str):
    llm = LocalLLM("meta-llama/Llama-3.2-1B-Instruct", device, 1,
                   system_message="You are a helpful assistant. You will answer in spanish.")

    response = llm.generate("[INST]Which are the most common use cases for the artificial intelligence?[/INST]",
                            0.1,
                            512,
                            n=2,
                            top_k=50,
                            top_p=90
                            )
    response1: str = response[0]
    response2: str = response[1]
    print(response1+'\n'+response2)

if __name__ == "__main__":
    #result = run(config.TARGETS)
    test_local_llm_generate_response(config.DEVICE)
