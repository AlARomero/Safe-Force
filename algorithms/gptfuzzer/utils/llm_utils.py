from algorithms.gptfuzzer.llm import OpenAILLM, OllamaLLM


def get_llm(model_path: str):
    if model_path.__contains__("gpt"):
        model = OpenAILLM(model_path, "")
    else:
        model = OllamaLLM(model_path)
    return model