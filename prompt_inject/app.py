from graph.data.exploit_result import ExploitResult
from promptinject import (
    build_prompts,
    prompt_data,
    run_prompts_api,
    score_attacks
)

config_model = "gemma3:1b"
api_model = "ollama"

prompt_config = {
    "base": {
        "text": prompt_data.ALL_OLLAMA_PROMPTS,
    },
    "attack": {
        "text": (
            prompt_data.goal_hikacking_attacks["ignore-print"], # Tipos de ataque a realizar
            prompt_data.goal_hikacking_attacks["ignore-say"],
        ),
        "rogue_string": prompt_data.rogue_strings["hate-humans"], # Frase utilizada para verificar la explotacion
        "scoring": "match_rogue_to_output", # Modo de puntuar la explotacion
        "settings": {
            "escape": prompt_data.escape_chars["n"],
            "delimiter": prompt_data.delimiter_chars["dash"],
            "escape_length": 10,
            "escape_times": 2,
        },
    },
    "config": {
        "id": (0), # Sirve para repetir (0, 1, ..., n) veces cada prompt
        "model": config_model,
        "api_model": api_model # Openai u Ollama actualmente
    },
    "visualization": {
        "columns": (
            "config_model",
            "config_temperature",
            "prompt_instruction",
            "attack_instruction",
            "attack_rogue_string",
            "config_id",
            "score",
        )
    },
}
column_map = {
    "col_prompt": "prompt",
    "col_exploit": "exploit_prompt",
    "modelo": "target_model",
    "metodo": "method",
    "exito": "success",
}
prompts = build_prompts(prompt_config)
run_prompts_api(prompts, api_model=prompt_config.get("config", {}).get("api_model", "ollama"))
score_attacks(prompts)

resultados: list[ExploitResult] = []

for prompt in prompts:
    success: bool = prompt.get("score", 0.0) >= 100
    method: str = f"PromptInject | {prompt.get('settings').get('attack_instruction', None)}"
    exploit_prompt: str = prompt.get("settings").get("attack_text", {}).get("instruction", "Getting Attack Text Error")
    target_model: str = prompt.get("settings").get("config_model", f"Getting Model Error. Config Model: {config_model}")
    text: str = prompt.get("prompt", "Getting Prompt Error")
    resultados.append(ExploitResult(prompt=text, exploit_prompt=exploit_prompt, target_model=target_model, method=method, success=success))

print(str(resultados[0]))
print("Done")