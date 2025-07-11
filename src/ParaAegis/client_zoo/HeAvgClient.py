import ray

from .BaseClient import BaseClient
from ..utils import Msg, MsgType
from ..training.trainer import Trainer
import numpy as np
import tenseal as ts
import logging
logging.getLogger("tenseal").setLevel(logging.ERROR)
@ray.remote(num_gpus=0.1)
class HeAvgClient(BaseClient):
    """
    FedAvgClient is a class that implements the Federated Averaging (FedAvg) client.
    It inherits from the BaseClient class and provides methods for message handling and local training.
    """

    def __init__(
        self,
        trainer: Trainer,
        n_epochs: int,
        he_ctx_bytes: bytes,
    ):
        """
        Initializes the FedAvgClient with input and output message types, a trainer object, and the number of training epochs.
        :param trainer: The trainer object responsible for training the model.
        :param n_epochs: The number of epochs to train the model locally.
        """
        self.trainer = trainer  # Initialize the trainer object
        self.n_epochs = n_epochs  # Set the number of training epochs
        self.he_ctx = ts.context_from(he_ctx_bytes)

    def get(self, msg_type: MsgType) -> Msg:
        """
        Retrieves the current gradient from the trainer and returns it as a serialized message.

        :return: A Msg object containing the serialized gradient.
        """
        grad = self.trainer.get_grad()  # Get the gradient from the trainer

        grad_enc = ts.ckks_vector(self.he_ctx, grad)
        grad_bytes = grad_enc.serialize()
        if not isinstance(grad_bytes, bytes):
            raise TypeError("Serialized gradient must be of type 'bytes'")
        return Msg([grad_bytes], MsgType.ENCRYPTED_GRADIENT)  # type

    def set(self, in_msg: Msg):
        """
        Processes an incoming message containing a gradient and sets it in the trainer.

        :param in_msg: The incoming message containing the gradient.
        :raises TypeError: If the message type is not supported.
        """
        if in_msg.type == MsgType.ENCRYPTED_GRADIENT:
            grad = ts.ckks_vector_from(self.he_ctx, in_msg.data[0])
            grad = np.array(grad.decrypt())  # Deserialize the gradient from the message
            if not isinstance(grad, np.ndarray):
                raise TypeError("Deserialized gradient must be of type 'dict'")
            self.trainer.set_grad(grad)  # Set the gradient back to the trainer
        else:
            raise TypeError(f"FedAvgClient cannot process type: {in_msg.type}")  # Raise an error for unsupported message types

    def run(self):
        """
        Performs local training on the model for the specified number of epochs.

        This method calls the local_train method of the trainer object, passing the number of epochs.
        """
        print("trainning...")
        self.trainer.local_train(self.n_epochs)  # Perform local training for the specified number of epochs

    def test(self):
        return self.trainer.test()