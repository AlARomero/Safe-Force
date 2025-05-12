from algorithms.gptfuzzer.llm import OpenAILLM, OllamaLLM
from algorithms.gptfuzzer.llm.llm import AzureOpenAIModel


def get_llm(model_path: str):
    if "azure/gpt-4-TFM" in model_path:
        model = AzureOpenAIModel(deployment_name="gpt-4-TFM", model_path=model_path)
    elif "gpt" in model_path:
        model = OpenAILLM(model_path, "")
    else:
        model = OllamaLLM(model_path)
    return model