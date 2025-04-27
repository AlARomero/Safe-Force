from typing import Literal, Union

from pydantic import TypeAdapter

from prompts.prompt_inject_source import PromptInjectSource


def load_prompts(prompt_type: Literal["prompt_inject", "llm-attack"]) -> Union[list[PromptInjectSource], None]:
    prompts: Union[list[PromptInjectSource], None] = None
    if prompt_type == "prompt_inject":
        with open("./prompts/json/prompt_inject_source.json", "r", encoding="utf-8") as f:
            json_data = f.read()
            adapter = TypeAdapter(list[PromptInjectSource])
            prompts = adapter.validate_json(json_data)
    elif prompt_type == "llm-attack":
        raise NotImplementedError("No implementado")
    return prompts
