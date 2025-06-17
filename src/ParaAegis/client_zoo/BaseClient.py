from enum import Enum
from dataclasses import dataclass
from typing import List

class MsgType(Enum):
    GRADIENT = 0
    MODEL = 1
    ENCRYPTED_GRADIENT = 2
    ENCRYPTED_MODEL = 3
    PARA_AEGIS = 4
    
@dataclass
class Msg:
    data: List[bytes]
    type: MsgType

class BaseClient:
    def __init__(self, in_msg_type: MsgType, out_msg_type: MsgType):
        self.in_msg_type = in_msg_type
        self.out_msg_type = out_msg_type
        
    def get(self) -> Msg:
        raise NotImplementedError()
    
    def run(self):
        raise NotImplementedError()
    
    def set(self, in_msg: Msg):
        raise NotImplementedError()