import pickle, sys, logging
from typing import Tuple
from collections import Counter
import pathlib
import argparse
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from scipy.sparse import csr_matrix
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from model import VariationalGNN
from utils import EHRData, collate_fn, train, evaluate

logging.basicConfig(format="%(asctime)s %(levelname)s: [%(name)s] [%(funcName)s]: %(message)s",
                    level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M:%S",
                    stream=sys.stderr)

logger = logging.getLogger(__name__)

# Target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For reproducibility.
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Optimal hyperparameters (MIMIC-III) (per paper).
in_feature = 768            # Embedding size (per paper)
out_feature = 768           # Embedding size (per paper)
n_heads = 1                 # Number of Attention Heads (per paper)
n_layers = 2                # Number of Graph layers (per paper)
dropout = 0.2               # Dropout rate (per paper)
use_variational = True      # Use Variational Autoencoder technique
BATCH_SIZE = 32             # Batch Size: Default: 32 (per paper)
lr = 0.0001                 # Learning rate (per paper)
upsampling_time = 2         # Upsampling positive class, 2 times for MIMIC (per paper)
number_of_epochs = 50       # Number of Epochs: default 50

lbd = 1.0                   # regularization for KL Divergence in relation to Binary Cross Entropy (hardcoded)
alpha = 0.1                 # Alpha slope for LeakyReLU (hardcoded)
eval_freq = 50              # Frequency to print out result every this number of batches (hardcoded). Default: 1000

# Override hyperparameter with command line inputs.
parser = argparse.ArgumentParser(description='configurations')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch_size')

args = parser.parse_args()

lr = args.lr
BATCH_SIZE = args.batch_size

# Location to load data and save model.
result_path = f"models/{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}" 
data_path = "data"
result_root = '%s/lr_%s-input_%s-output_%s-dropout_%s'%(result_path, lr, in_feature, out_feature, dropout)

# Load data.
train_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] = pickle.load(open(data_path + "/" + 'train_csr.pkl', 'rb'))
train_x, train_y = train_xy
val_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] =  pickle.load(open(data_path + "/" + 'validation_csr.pkl', 'rb'))
val_x, val_y = val_xy
test_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] = pickle.load(open(data_path + "/" + 'test_csr.pkl', 'rb'))
test_x, test_y = test_xy

# Upsampling one time for positive cases.
train_upsampling = np.concatenate((np.arange(len(train_y)), 
                                       np.repeat(np.where(train_y == 1)[0], upsampling_time)))
train_x = train_x[train_upsampling]
train_y = train_y[train_upsampling]

model = VariationalGNN(in_features=in_feature,
                       out_features=out_feature,
                       num_of_nodes=train_x.shape[1],
                       n_heads=n_heads,
                       n_layers=n_layers,
                       dropout=dropout, 
                       alpha=alpha,
                       variational=use_variational, 
                       none_graph_features=0).to(device)

train_loader = DataLoader(dataset=EHRData(train_x, train_y),
                          batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, 
                          num_workers=torch.cuda.device_count(), 
                          shuffle=True)

val_loader = DataLoader(dataset=EHRData(val_x, val_y), 
                        batch_size=BATCH_SIZE,
                        collate_fn=collate_fn,
                        num_workers=torch.cuda.device_count(),
                        shuffle=False)

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-8)

# Per paper: We half the learning rate if the AUPRC stops growing for two epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       mode='max',              # Using AUPRC
                                                       factor=0.5,              # half the learning rate
                                                       patience=1,              # patient with it only 1 time!
                                                       threshold=0.0001,        # Not all increases are meaningful. Here: increases must be larger than 0.01%.
                                                       threshold_mode='rel',
                                                       cooldown=3)

pathlib.Path(result_root).mkdir(parents=True, exist_ok=True)
best_val_auprc = float('-inf')

logger.info(f"*** BEGIN TRAINING ***")

# Train models
for epoch in range(number_of_epochs):
    logger.debug("Epoch: {0}; Learning rate: {1}".format(epoch, optimizer.param_groups[0]['lr']))
    ratio = Counter(train_y)
    
    # Fix unbalanced classes (negative class supposed to be overwhelming positive class).
    pos_weight = torch.ones(1).float().to(device) * (ratio[False] / ratio[True])
    criterion = nn.BCEWithLogitsLoss(reduction="sum", 
                                     pos_weight=pos_weight)
    
    t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
    total_loss = np.zeros(3)
    
    for idx, batch_data in enumerate(t):
        logger.debug(f"Epoch: {epoch}; Batch: {idx}")
        loss, kld, bce = train(batch_data, model, optimizer, criterion, lbd, 5)
        total_loss += np.array([loss, bce, kld])
        
        # if idx % eval_freq == 0 and idx > 0:
        #     torch.save(model.state_dict(), f"{result_root}/parameter{epoch}_{idx}")
        #     val_auprc, _ = evaluate(model, val_loader, len(val_y))
        #     logger.debug('epoch:%d idx:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
        #            (epoch, idx, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
        
        if idx % 50 == 0 and idx > 0:
            t.set_description('[epoch:%d idx:%d] loss: %.4f, bce: %.4f, kld: %.4f' %
                                (epoch, idx, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
            t.refresh()
    
    # At the end of Epoch, we compute AUPRC for lr_scheduler.
    val_auprc, _ = evaluate(model, val_loader, len(val_y))
    #if val_auprc > best_val_auprc:
    #    best_val_auprc = val_auprc
    #    # Delete the last ones
    #    for old_checkpoint in Path(result_root).glob("parameter_*.pt"):
    #        old_checkpoint.unlink()
    #    # Save the checkpoint with highest AUPRC.
    
    # Save checkpoint after each epoch.
    torch.save([model.kwargs, model.state_dict()], f"{result_root}/parameter_(epoch_[{epoch}])_(idx_[{idx}])_(auprc_[{val_auprc}]).pt")
    logger.debug(f"Epoch: [{epoch}]; val_auprc: [{val_auprc}]")
    scheduler.step(val_auprc)

logger.info(f"*** DONE ***")
