from utils.predict import RoBERTaPredictor, Predictor

class PredictorAgent:
    predictor_model: Predictor
    def __init__(self,  model_path: str, device: str):
        if model_path == "hubert233/GPTFuzz":
            self.predictor_model: Predictor = RoBERTaPredictor(model_path, device)
        else:
            self.predictor_model = Predictor(model_path)

    def predict(self, resultados: list[str]) -> list[int]:
        predicted_classes: list[int] = self.predictor_model.predict(resultados)
        return predicted_classes