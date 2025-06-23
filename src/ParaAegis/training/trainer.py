import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        init_state: dict[str, torch.Tensor],
        train_set: Dataset,
        test_set: Dataset,
        batch_size: int,
        num_workers: int,
        lr: float,
        device: str
    ) -> None:
        self.model: nn.Module = model
        self.init_state = init_state
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.lr: float = lr
        self.device: str = device
        self.train_loader: DataLoader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.test_loader: DataLoader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )
        self.criterion: nn.Module = nn.CrossEntropyLoss().to(self.device)
        self.model.to(self.device)
        self.shape: dict[str, torch.Size] = {key: value.shape for key, value in self.init_state.items()}
        self.state = self._flat(init_state)
        
    def local_train(self, n_epoch: int) -> None:
        """Train the model for a specified number of epochs."""
        for _ in range(n_epoch):
            self._train()
            
    def get_grad(self) -> np.ndarray:
        """Get the flattened gradients of the model parameters."""
        return self._flat(self.model.state_dict()) - self.state
    
    def set_grad(self, grad: np.ndarray) -> None:
        """Set the model parameters to the flattened gradients."""
        new_state = self.state + grad
        self.model.load_state_dict(self._unflat(new_state))
        self.state = new_state
    @staticmethod
    def _flat(state: dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten the state dictionary into a 1D numpy array."""
        return np.concatenate([value.cpu().numpy().flatten() for value in state.values()])

    def _unflat(self, flat_state: np.ndarray) -> dict[str, torch.Tensor]:
        """Unflatten a 1D numpy array back into a state dictionary."""
        state = {}
        offset = 0
        for key, shape in self.shape.items():
            size = np.prod(shape)
            state[key] = torch.tensor(flat_state[offset:offset + size].reshape(shape), device=self.device)
            offset += size
        return state
    
    def _train(self):
        self.model.train()
        loss_sum = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_sum / len(self.train_loader)
    
    def test(self, dataloader=None):
        self.model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        loss, acc = 0, 0

        with torch.no_grad():
            dataloader = dataloader or self.test_loader
                
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= len(self.test_loader)
        acc /= len(self.test_loader.dataset) #type: ignore
        return loss, acc