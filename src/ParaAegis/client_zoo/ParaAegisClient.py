from .BaseClient import BaseClient
from ..utils.msg import MsgType, Msg
from ..training.trainer import Trainer
import ray
import tenseal as ts
import numpy as np


@ray.remote
class ParaAegisClient(BaseClient):

    def __init__(
    self,
    trainer: Trainer,
    n_epochs: int,
    he_ctx_bytes: bytes,
    partition_config: dict[str, str],
    ):
        self.trainer = trainer  # Initialize the trainer object
        self.n_epochs = n_epochs  # Set the number of training epochs
        self.he_ctx = ts.context_from(he_ctx_bytes)
        self.partition = partition_config  # Set the partition configuration
        self.sgn_history = np.array([])  # Initialize the sign history as an empty array
        self.grad = np.array([])  # Initialize the gradient as an empty array

    def get(self, msg_type: MsgType) -> Msg:

        if msg_type == MsgType.PARTIAL_ENCRYPTED_GRADIENT:
            pass
        elif msg_type == MsgType.VOTE:
            pass
        else:
            raise TypeError(f"Unknown msg_type: {msg_type}")



    def set(self, in_msg: Msg):
        pass

    def run(self):
        """
        Performs local training on the model for the specified number of epochs.

        This method calls the local_train method of the trainer object, passing the number of epochs.
        """
        self.trainer.local_train(self.n_epochs)  # Perform local training for the specified number of epochs
        self.grad = self.trainer.get_grad()  # Retrieve the gradient after training
        self.sgn_history += np.sign(self.grad)  # Update the sign history with the current gradient's sign

    def _get_he_idcs(self) -> np.ndarray:
        he_idcs = np.argwhere(np.sign(self.grad) == self.sgn_history)  # Get indices where the sign matches the history
        return he_idcs
