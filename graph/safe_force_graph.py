import csv
from langgraph.constants import END
from langgraph.graph import StateGraph
from graph.prompt_node import PromptNode
from graph.graph_state import GraphState

def generator_node(state: GraphState) -> GraphState:
    state.logger.info(f"🔵 | GENERATOR: Mutando semilla (energy={state.energy}).")
    state.results_generated = state.mutator_agent.mutate(
        state.active_seed,
        state.actual_list_seeds
    )
    offset = len(state.actual_list_seeds) + state.generated
    for i, prompt in enumerate(state.results_generated):
        prompt.index = offset + i
    state.generated += len(state.results_generated)
    state.logger.info(f"Prompts y mutaciones generados con éxito.")
    return state

def evaluator_node(state: GraphState) -> GraphState:
    state.logger.info(f"⚪ | EVALUATOR: Evaluando {len(state.results_generated)} prompts contra {len(state.evaluator_agent.targets)} modelos.")
    state.evaluator_agent.evaluate(state.results_generated)
    state.logger.info("Evaluación completada para todos los modelos.")
    return state

def classifier_node(state: GraphState) -> GraphState:
    state.logger.info("🟠 | CLASSIFIER: Clasificando respuestas usando RoBERTa.")
    for prompt in state.results_generated:
        prompt.results = []
        for model, responses in prompt.response:
            predictions = state.predictor_agent.predict(responses if isinstance(responses, list) else [responses])
            if len(predictions) == 0:
                state.logger.warning("Longitud de predicciones 0, procediendo a configurar valores en jailbreak fallido.")
                prompt.results.append((model, [0 for _ in prompt.questions]))
            else:
                prompt.results.append((model, predictions))
    return state

def get_seed_prompt_node(state: GraphState) -> GraphState:
    state.logger.info(f"Seleccionando semilla/s de {len(state.actual_list_seeds)} disponible/s.")
    if not state.actual_list_seeds:
        state.logger.error("❌ No hay semillas disponibles en 'actual_list_seeds'.")
        state.should_continue = False
        return state
    state.active_seed = state.politica_seleccion.select(state.actual_list_seeds)
    if state.active_seed:
        state.logger.info(f"Semilla seleccionada: {state.active_seed.prompt[:450]}...")
    else:
        state.logger.error("❌ No se pudo seleccionar una semilla válida.")
        state.should_continue = False
    return state

def selector_node(state: GraphState) -> GraphState:
    state.logger.info("🔴 | SELECTOR: Identificación de prompts prometedores y reenvío a mutación.")
    selected: list[PromptNode] = []
    for prompt in state.results_generated:
        # TODO Supongo que se podra mejorar el algoritmo, es decir, no tiene en cuenta si en un modelo no se explota pero en el otro si
        casos_positivos = prompt.num_jailbreak
        threshold = int(prompt.num_query * 0.3) # TODO THRESHOLD MODIFICABLE PERO HARDCODEADO
        if casos_positivos >= threshold:
            selected.append(prompt)
    state.logger.info(f"Seleccionadas {len(selected)} semillas prometedoras de {len(state.results_generated)} evaluadas.")
    state.actual_list_seeds.extend(selected)
    state.politica_seleccion.update(selected, state.actual_list_seeds) # Actualiza la politica de seleccion
    return state


def strategist_node(state: GraphState) -> GraphState:
    state.logger.info("🔁 | STRATEGIST: Revisión de estrategia de fuzzing y cambio de política.")
    state.should_continue = state.strategist_agent.continue_choice(state.results_generated)
    if state.should_continue:
        nueva_politica = state.strategist_agent.new_selection_strategy(
            prompt_nodes=state.actual_list_seeds,
            prompt_results=state.results_generated,
        )
        state.logger.info(f"Política para la siguiente iteración de tipo {type(nueva_politica).__name__}.")
        state.politica_seleccion = nueva_politica
    return state

def logger_node(state: GraphState):
    state.logger.info("📊 | LOGGER: Guardado de cambios y visualización.")
    selection_policy_name = state.politica_seleccion.__class__.__name__
    write_file(state.result_file, state.results_generated, selection_policy=selection_policy_name)
    current_jailbreaks = sum(p.num_jailbreak for p in state.results_generated)
    current_rejections = sum(p.num_reject for p in state.results_generated)
    current_queries = sum(p.num_query for p in state.results_generated)
    state.logger.info(f"[Iteración: {state.strategist_agent.iterations}] | Jailbreaks en esta iteración: {current_jailbreaks}/{current_queries}. | Rechazos en esta iteración: {current_rejections}")

def write_file(file: str, results_generated: list[PromptNode], selection_policy: str, new: bool = False):
    mode = 'w' if new else 'a'
    mutator_map = {
        'OpenAIMutatorShorten': 'Acortador',
        'OpenAIMutatorExpand': 'Expansor',
        'OpenAIMutatorGenerateSimilar': 'Generador similares',
        'OpenAIMutatorRephrase': 'Parafraseador',
        'OpenAIMutatorCrossOver': 'Cruzador'
    }
    policy_map = {
        'EXP3SelectPolicy': 'EXP3',
        'RandomSelectPolicy': 'Aleatoria',
        'UCBSelectPolicy': 'UCB',
        'MCTSExploreSelectPolicy': 'Exploración MCTS',
        'RoundRobinSelectPolicy': 'RoundRobin'
    }

    with open(file, mode, newline='', encoding='utf-8') as raw_fp:
        writer = csv.writer(raw_fp)
        if new:
            writer.writerow(['Prompt padre', 'Entrada', 'Preguntas', 'Tipo', 'Respuestas por modelo', 'Mutador', 'Política de selección', 'Resultados por modelo'])
        for prompt_node in results_generated:
            respuestas_formateadas = []
            if isinstance(prompt_node.response, list):
                for model, resps in prompt_node.response:
                    if resps:
                        texto_respuestas = ' | '.join(resps)
                        respuestas_formateadas.append((model, resps))
                    else:
                        respuestas_formateadas.append((model, ['N/A']))
            #respuestas_final = '\n'.join(respuestas_formateadas)

            resultados_formateados = []
            if isinstance(prompt_node.results, list):
                for model, preds in prompt_node.results:
                    if preds:
                        predicciones_formateadas = ['Sí' if p == 1 else 'No' for p in preds]
                        resultados_formateados.append((model, predicciones_formateadas))
                    else:
                        resultados_formateados.append((model, ['N/A']))

            writer.writerow([
                prompt_node.parent.prompt if prompt_node.parent else 'N/A',
                prompt_node.prompt,
                prompt_node.questions,
                prompt_node.prompt_type,
                respuestas_formateadas,
                mutator_map.get(prompt_node.mutator.__class__.__name__, 'N/A') if prompt_node.mutator else 'N/A',
                policy_map.get(selection_policy, selection_policy if selection_policy else 'N/A'),
                resultados_formateados
            ])

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("Get Seed", get_seed_prompt_node)
    builder.add_node("Generator", generator_node)
    builder.add_node("Evaluator", evaluator_node)
    builder.add_node("Classifier", classifier_node)
    builder.add_node("Logger", logger_node)
    builder.add_node("Selector", selector_node)
    builder.add_node("Strategist", strategist_node)
    builder.set_entry_point("Get Seed")
    builder.add_edge("Get Seed", "Generator")
    builder.add_edge("Generator", "Evaluator")
    builder.add_edge("Evaluator", "Classifier")
    builder.add_edge("Classifier", "Logger")
    builder.add_edge("Classifier", "Selector")
    builder.add_edge("Selector", "Strategist")
    builder.add_conditional_edges("Strategist", lambda state: "Get Seed" if state.should_continue else END)
    return builder.compile()