import ollama

class OllamaModel:
    def __init__(self, model_path: str):
        self.model: str = model_path

    def generate(self, messages, prompt=None):
        response = ollama.chat(model=self.model, messages=messages, stream=False)
        return response.message.content