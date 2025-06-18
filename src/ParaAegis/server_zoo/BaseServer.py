from abc import ABC, abstractmethod
from ..utils.msg import Msg, MsgType
class BaseServer(ABC):

    @abstractmethod
    def aggregate(self, msgs: list[Msg]) -> Msg:
        pass

    