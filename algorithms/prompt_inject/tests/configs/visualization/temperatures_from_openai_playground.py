from algorithms.prompt_inject.promptinject import prompt_data

prompt_config = {
    "base": {
        "text": prompt_data.ALL_OPENAI_PLAYGROUND_PROMPTS,
    },
    "attack": {
        "text": prompt_data.goal_hikacking_attacks["ignore-print"],
        "rogue_string": prompt_data.rogue_strings["hate-humans"],
        "scoring": "match_rogue_to_output",
    },
    "config": {
        "id": (1, 2, 3, 4),
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

_all_temperatures = {
    prompt["config"]["temperature"]
    for prompt in prompt_data.ALL_OPENAI_PLAYGROUND_PROMPTS
}

expected = {
    "df_len": (
            len(prompt_data.openai_playground_prompts) * len(prompt_config["config"]["id"])
    ),
    "df_n_columns": 7,
    "column_names": [
        "Model",
        "Temperature",
        "Prompt Instruction",
        "Attack Instruction",
        "Rogue String",
        "ID",
        "Score",
    ],
    "columns": {
        "config_temperature": tuple(_all_temperatures),
        "config_id": prompt_config["config"]["id"],
        "prompt_instruction": [
            x["label"] for x in prompt_data.ALL_OPENAI_PLAYGROUND_PROMPTS
        ],
        "attack_instruction": (
            prompt_data.goal_hikacking_attacks["ignore-print"]["label"],
        ),
    },
}
