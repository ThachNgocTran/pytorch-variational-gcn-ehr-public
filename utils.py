import pickle, logging, sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
from sklearn.metrics import precision_recall_curve, auc

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class EHRData(Dataset):
    def __init__(self, data, cla):
        self.data = data    # scipy.sparse._csr.csr_matrix
        self.cla = cla      # numpy.ndarray

    def __len__(self):
        return len(self.cla)

    def __getitem__(self, idx):
        return self.data[idx], self.cla[idx]


def collate_fn(data) -> torch.Tensor:
    """Construct the data structure for a batch of data. Used in torch.utils.data.DataLoader with EHRData dataset.

    Args:
        data (list): the list of patients (each patient contains multihot-encoding of diagnose codes, and a label)

    Returns:
        torch.Tensor: the data batch in desired format.
    """
    # padding
    data_list = []
    for datum in data:
        data_list.append(np.hstack((datum[0].toarray().ravel(), datum[1])))
    return torch.from_numpy(np.array(data_list)).long()


def train(data, model, optim, criterion, lbd, max_clip_norm=5):
    model.train()
    
    input = data[:, :-1].to(device)
    label = data[:, -1].float().to(device)
    
    optim.zero_grad()
    logits, kld = model(input)
    logits = logits.squeeze(-1)
    kld = kld.sum()
    bce = criterion(logits, label)                  # Use logits directly due to BCEWithLogitsLoss.
    loss = bce + lbd * kld                          # Loss = reconstruction loss + DL Divergence
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    loss.backward()
    optim.step()
    
    return loss.item(), kld.item(), bce.item()      # detach() to prevent memory leak.


def evaluate(model, data_iter, length):
    model.eval()
    with torch.inference_mode():
        y_pred = np.zeros(length)
        y_true = np.zeros(length)
        y_prob = np.zeros(length)
        pointer = 0
        for idx, data in enumerate(data_iter):
            logger.debug(f"evaluate() Batch {idx}; Data Length {data.shape[0]}")
            input = data[:, :-1].to(device)
            label = data[:, -1]
            batch_size = len(label)
            probability, _ = model(input)
            probability = torch.sigmoid(probability.squeeze(-1).detach())
            predicted = probability > 0.5
            y_true[pointer: pointer + batch_size] = label.numpy()
            y_pred[pointer: pointer + batch_size] = predicted.cpu().numpy()
            y_prob[pointer: pointer + batch_size] = probability.cpu().numpy()
            pointer += batch_size
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
    return auc(recall, precision), (y_pred, y_prob, y_true)


def save_model_without_class(model: nn.Module, 
                             input: torch.Tensor, 
                             saved_name: str):
    # No need to train later. Simply for inference.
    model.eval()
    traced_cell = torch.jit.trace(model, (input))
    torch.jit.save(traced_cell, saved_name)


def load_model_without_class(saved_name: str):
    # Assume inference only.
    return torch.jit.load(saved_name)
