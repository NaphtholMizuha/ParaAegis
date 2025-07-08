from .BaseClient import BaseClient
from ..utils.msg import MsgType, Msg
from ..training.trainer import Trainer
import ray
import tenseal as ts
import numpy as np
from typing import Tuple


@ray.remote(num_gpus=0.1)
class ParaAegisClient(BaseClient):

    def __init__(
    self,
    trainer: Trainer,
    n_epochs: int,
    he_ctx_bytes: bytes,
    he_fraction: float = 0.1,
    noise_multiplier: float = 1.0,
    clip_thr: float = 1.0
    ):
        self.trainer = trainer  # Initialize the trainer object
        self.n_epochs = n_epochs  # Set the number of training epochs
        self.he_ctx = ts.context_from(he_ctx_bytes)
        self.grad = np.array([])  # Initialize the gradient as an empty array
        self.he_fraction = he_fraction
        self.global_votes = None  # Initialize global votes as None
        self.noise_multiplier = noise_multiplier
        self.clip_thr = clip_thr

    def get(self, msg_type: MsgType) -> Msg:

        if msg_type == MsgType.PARTIAL_ENCRYPTED_GRADIENT:
            he_part, dp_part = self._get_grad(self.grad)  # Get the gradient and partition it
            return Msg([he_part, dp_part], msg_type)
        elif msg_type == MsgType.VOTE:
            vote = self._get_votes()
            return Msg([vote.tobytes()], msg_type)  # Return the vote from the trainer
        else:
            raise TypeError(f"Unknown msg_type: {msg_type}")



    def set(self, in_msg: Msg):
        if in_msg.type == MsgType.PARTIAL_ENCRYPTED_GRADIENT:
            he_bytes, dp_bytes = in_msg.data  # Extract the HE and DP parts from the message
            he_part = ts.ckks_vector_from(self.he_ctx, he_bytes)  # Deserialize
            he_part = np.array(he_part.decrypt())  # Decrypt the HE part
            grad = np.frombuffer(dp_bytes, dtype=np.float32).copy()  # Deserialize the
            grad[self.global_votes] = he_part  # Set the HE part in the gradient
            self.trainer.set_grad(grad)  # Set the gradient back to the trainer
        elif in_msg.type == MsgType.VOTE:
            self.global_votes = np.frombuffer(in_msg.data[0], dtype=np.int32)  # Deserialize the global votes from the message
        else:
            raise TypeError(f"ParaAegisClient cannot process type: {in_msg.type}")

    def run(self):
        """
        Performs local training on the model for the specified number of epochs.

        This method calls the local_train method of the trainer object, passing the number of epochs.
        """
        self.trainer.local_train(self.n_epochs)  # Perform local training for the specified number of epochs
        self.grad = self.trainer.get_grad()  # Retrieve the gradient after training

    def _get_votes(self) -> np.ndarray:
        he_len = int(len(self.grad) * self.he_fraction)
        he_idcs = np.argpartition(np.abs(self.grad), he_len)[:he_len]  # Get indices of the top he_len elements
        return he_idcs.astype(np.int32)  # Convert indices to int32 for serialization
    
    def _get_grad(self, grad: np.ndarray) -> Tuple[bytes, bytes]:
        he_part, dp_part = self._partition_grad(grad)  # Partition the gradient into HE and DP parts
        he_part = self._encrypt_grad(he_part)  # Encrypt the HE part of the gradient
        dp_part = self._perturb_grad(dp_part)  # Perturb the DP part
        return he_part, dp_part

    def _partition_grad(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        he_part = grad[self.global_votes]
        dp_part = grad.copy()
        dp_part[self.global_votes] = 0  # Set the global votes indices to zero
        return he_part, dp_part
    
    def _perturb_grad(self, grad: np.ndarray) -> bytes:
        """
        Perturbs the gradient by adding Gaussian noise.
        :param grad: The gradient to perturb.
        :return: The perturbed gradient as bytes.
        """
        grad *= np.min([1.0, float(self.clip_thr / np.linalg.norm(grad))])
        grad += np.random.normal(0, self.noise_multiplier, size=grad.shape)
        return grad.tobytes()

    def _encrypt_grad(self, grad: np.ndarray) -> bytes:
        """
        Encrypts the gradient using the HE context.
        :param grad: The gradient to encrypt.
        :return: The encrypted gradient as bytes.
        """
        grad_enc = ts.ckks_vector(self.he_ctx, grad)
        grad_bytes = grad_enc.serialize()
        if not isinstance(grad_bytes, bytes):
            raise TypeError("Serialized gradient must be of type 'bytes'")
        return grad_bytes
    
    def test(self):
        return self.trainer.test()