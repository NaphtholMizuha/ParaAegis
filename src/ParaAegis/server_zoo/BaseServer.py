from abc import ABC, abstractmethod
class BaseServer(ABC):

    @abstractmethod
    def aggregate(self):
        pass

    