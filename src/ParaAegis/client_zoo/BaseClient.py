from ..utils.msg import MsgType, Msg
from abc import ABC, abstractmethod


class BaseClient(ABC):

    @abstractmethod
    def get(self, msg_type: MsgType) -> Msg:
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @abstractmethod
    def set(self, in_msg: Msg):
        raise NotImplementedError()