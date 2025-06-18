from dataclasses import dataclass
from enum import Enum
from typing import List


class MsgType(Enum):
    GRADIENT = 'gradient'
    ENCRYPTED_GRADIENT = 'encrypted_gradient'
    PARTIAL_ENCRYPTED_GRADIENT = 'partial_encrypted_gradient'
    VOTE = 'vote'


@dataclass
class Msg:
    data: List[bytes]
    type: MsgType
