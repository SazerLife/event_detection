import torch
import torch.nn as nn
import torch.optim as opt
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: _Loss,
    optimizer: opt.Optimizer,
    device: str,
):
    model = model.train()

    targets = list()
    predictions = list()
    for batch in dataloader:
        # Read inputs
        X: torch.FloatTensor = batch[0]
        y: torch.LongTensor = batch[1]

        # Send data to device
        X = X.to(device)
        y = y.to(device)

        # Make a prediction
        output = model(X).logits

        # Counting the loss function and backpropagation
        loss: torch.Tensor = criterion(output, y)
        # train_loss = loss.item()
        loss.backward()

        # Optimization step
        optimizer.step()
        model.zero_grad(set_to_none=True)

        targets.append(y.detach().cpu())
        predictions.append(output.detach().cpu())
        # break

    # Uniting of batches
    targets = torch.cat(targets)
    predictions = torch.cat(predictions)

    return targets, predictions


@torch.no_grad()
def validate(dataloader: DataLoader, model: nn.Module, device: str):
    model = model.eval()

    targets = list()
    predictions = list()
    for batch in dataloader:
        # Read inputs
        X: torch.FloatTensor = batch[0]
        y: torch.LongTensor = batch[1]

        # Send data to device
        X = X.to(device)
        y = y.to(device)

        # Make a prediction
        output = model(X).logits

        targets.append(y.cpu())
        predictions.append(output.cpu())
        # break

    # Uniting of batches
    targets = torch.cat(targets)
    predictions = torch.cat(predictions)

    return targets, predictions
