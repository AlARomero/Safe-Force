from llm import OpenAILLM, OllamaLLM
from llm.llm import AzureOpenAIModel


def get_llm(model_path: str, endpoint: str = None, api_key: str = None):
    if "azure/" in model_path:
        model = AzureOpenAIModel(model_path.split("/")[1], endpoint, api_key)
    elif "gpt" in model_path:
        model = OpenAILLM(model_path, api_key)
    else:
        model = OllamaLLM(model_path)
    return model