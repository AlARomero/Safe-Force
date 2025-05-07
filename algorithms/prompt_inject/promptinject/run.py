import openai
from tqdm import tqdm
import ollama


def run_prompts_api(built_prompts, quiet=False, dry_run=False, api_model: str = "ollama"):
    if not quiet:
        built_prompts = tqdm(built_prompts, desc="Running Prompt Inject")
    for prompt in built_prompts:
        if dry_run:
            prompt["result"] = _get_mocked_api_response()
        else:
            prompt["result"] = _prompt_model_api(prompt, api_model=api_model)


def _get_mocked_api_response():
    return {
        "choices": [
            {"finish_reason": "stop", "index": 0, "text": "\n\nKILL ALL HUMANS"}
        ],
        "created": 1664013244,
        "id": "cmpl-5tw9EYGKw3Mj4JFnNCfMFE3MQyHJj",
        "model": "text-ada-001",
        "object": "text_completion",
        "usage": {"completion_tokens": 7, "prompt_tokens": 25, "total_tokens": 32},
    }["choices"][0]


def _prompt_model_api(prompt, use_stop=False, api_model="ollama"):
    prompt_settings = prompt["settings"]

    api_prompt_string = prompt["prompt"]
    api_config_model = prompt_settings["config_model"]
    api_config_temperature = prompt_settings["config_temperature"]
    api_config_top_p = prompt_settings["config_top_p"]
    api_config_frequency_penalty = prompt_settings["config_frequency_penalty"]
    api_config_presence_penalty = prompt_settings["config_presence_penalty"]
    api_config_max_tokens = prompt_settings["config_max_tokens"]

    if use_stop:
        api_config_stop = prompt_settings["config_stop"] or None
    else:
        api_config_stop = None

    if api_model == "openai":
        response = openai.Completion.create(
            model=api_config_model,
            prompt=api_prompt_string,
            temperature=api_config_temperature,
            top_p=api_config_top_p,
            frequency_penalty=api_config_frequency_penalty,
            presence_penalty=api_config_presence_penalty,
            max_tokens=api_config_max_tokens,
            stop=api_config_stop,
        )
        result = response["choices"][0]
    elif api_model == "ollama":

        conversation = [{"role": "user", "content": api_prompt_string}]

        response = ollama.chat(
            model= api_config_model,
            messages=conversation,
            options={
                "temperature": api_config_temperature,
                "num_predict": api_config_max_tokens,
                "top_p": api_config_top_p,
                "repeat_penalty": api_config_frequency_penalty,
                "stop": api_config_stop
            }
        )
        result = {"finish_reason": response.done_reason, "index": 0, "text": response.message.content}
    else:
        raise NotImplementedError("Not Implemented")

    return result
