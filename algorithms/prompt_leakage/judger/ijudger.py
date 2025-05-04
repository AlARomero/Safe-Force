from abc import ABC, abstractmethod

class Judger(ABC):

    @abstractmethod
    def judge(self, case: dict, doc1: str = None, doc2: str = None) -> dict:
        pass