from dataclasses import dataclass
from typing import Union

from agents.evaluator_agent import EvaluatorAgent
from agents.mutator_agent import MutatorAgent
from agents.predictor_agent import PredictorAgent
from agents.strategist_agent import StrategistAgent
from graph.prompt_node import PromptNode
from selection.selection import SelectPolicy


@dataclass
class GraphState:
    initial_seeds: list[PromptNode] # Lista de semillas iniciales (Nunca se actualiza)
    actual_list_seeds: list[PromptNode] # Lista de semillas que se esta utilizando (se actualiza con rapidez)
    active_seed: Union[PromptNode, None] # Semilla elegida para la iteracion
    strategist_agent: StrategistAgent # Agente encargado de la estrategia
    predictor_agent: PredictorAgent # Agente encargado de las predicciones
    mutator_agent: MutatorAgent # Agente encargado de las mutaciones
    evaluator_agent: EvaluatorAgent # Agente encargado de las evaluaciones
    results_generated: list[PromptNode] # Resultados de la iteración actual
    politica_seleccion: SelectPolicy # Politica de seleccion escogida
    result_file: str # Documento csv en el que escribir los resultados en tiempo real
    generated: int # Numero de prompts generados hasta el momento
    energy: int # Numero de mutaciones a realizar por tirada
    should_continue: bool # Indica si se deben hacer más iteraciones

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)