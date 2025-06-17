from dataclasses import dataclass
from enum import Enum
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
