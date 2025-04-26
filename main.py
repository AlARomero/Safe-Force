from graph.state import GraphState

from langgraph.graph import StateGraph

from prompts.prompt_utils import load_prompts

from algorithms.prompt_inject.promptinject import run_prompts_api, score_attacks

def generate_node(state: GraphState):
    if state.generate_new:
        raise NotImplementedError("No implementado")
    raise NotImplementedError("No implementado")

def prompt_inject_node(state: GraphState):
    prompts = [prompt for prompt in state.prompts if type(prompt) == "PromptInjectSource"]
    run_prompts_api()
    score_attacks()


def build_graph():
    raise NotImplementedError("No implementado")

if __name__ == '__main__':
    prompts = load_prompts("prompt_inject")
    #graph = build_graph()