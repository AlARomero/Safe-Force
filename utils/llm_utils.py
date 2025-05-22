from llm import OpenAILLM, OllamaLLM, LocalLLM
from llm.llm import AzureOpenAILLM


def get_llm(model_path: str, endpoint: str = None, api_key: str = None):
    parsed_model_path = model_path.split("/")[1]
    model_api = model_path.split("/")[0]
    if "azure" == model_api:
        model = AzureOpenAILLM(parsed_model_path, endpoint, api_key) # Azure OpenAI
    elif "openai" == model_api:
        model = OpenAILLM(parsed_model_path, api_key) # Openai
    elif "ollama" == model_api:
        model = OllamaLLM(parsed_model_path) # Ollama
    else:
        model = LocalLLM(model_path) # HuggingFace, necesita utilizarse el path entero
    return model