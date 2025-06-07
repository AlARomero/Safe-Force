from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

from utils.logger import get_basic_logger


class Predictor:
    def __init__(self, path):
        self.path = path

    def predict(self, sequences):
        raise NotImplementedError("Predictor debe implementar un método de predicción.")


class RoBERTaPredictor(Predictor):
    def __init__(self, path, device='cuda'):
        super().__init__(path)
        self.device = device
        self.logger = get_basic_logger(self.__class__.__name__)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.path)

    def predict(self, sequences: list):
        if len(sequences) == 0:
            self.logger.warning("Intentando predecir secuencias vacías.")
            return []
        inputs = self.tokenizer(sequences, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        _, predicted_classes = torch.max(predictions, dim=1)
        predicted_classes = predicted_classes.cpu().tolist()
        return predicted_classes
