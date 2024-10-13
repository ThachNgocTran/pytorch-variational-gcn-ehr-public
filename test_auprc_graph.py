# Run models saved for each epoch, to get AUPRC on training, validation and testing data.
import pickle, logging, sys
from typing import Tuple
import glob
import re
from pathlib import Path
import pickle

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib.pylab import plt

from model import VariationalGNN
from utils import EHRData, collate_fn, evaluate

logging.basicConfig(format="%(asctime)s %(levelname)s: [%(name)s] [%(funcName)s]: %(message)s",
                    level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M:%S",
                    stream=sys.stderr)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def load_model(model_path: str) -> nn.Module:
    kwargs, state = torch.load(model_path, weights_only=False)
    model = VariationalGNN(**kwargs).to(device)
    model.load_state_dict(state)
    return model


def collect_auprc_data():
    """The model is supposed to be saved after each epoch, during training. Now we load each of these models, and run it through entire training, validation and testing data. The point is to eventually draw an AUPRC graph along the epochs.
    """
    data_path = "data"

    BATCH_SIZE = 32

    logger.info("*** START ***")

    # Data (train + validation + test)
    train_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] = pickle.load(open(data_path + "/" + 'train_csr.pkl', 'rb'))
    train_x, train_y = train_xy
    val_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] =  pickle.load(open(data_path + "/" + 'validation_csr.pkl', 'rb'))
    val_x, val_y = val_xy
    test_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] = pickle.load(open(data_path + "/" + 'test_csr.pkl', 'rb'))
    test_x, test_y = test_xy

    # DataLoaders (train + validation + test)
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

    test_loader = DataLoader(dataset=EHRData(test_x, test_y), 
                            batch_size=BATCH_SIZE,
                            collate_fn=collate_fn,
                            num_workers=torch.cuda.device_count(),
                            shuffle=False)

    # TO BE CHANGED HERE:
    path = r"path_to_models/parameter_*.*"
    saved_model_files = glob.glob(path)
    # Sample pattern: parameter_(epoch_[0])_(idx_[1520])_(auprc_[-inf]).pt
    file_pattern = r"parameter_\(epoch_\[(\d+)\]\)"

    if len(saved_model_files) == 0:
        raise Exception("No *.pt found in [{path}].")

    result = []

    # The saved model files may have an arbitrary order. ➔ Should sort based on epoch index.
    saved_model_files = natural_sort(saved_model_files)

    for idx, saved_model_file in enumerate(saved_model_files):
        # Extract the epoch index
        file_name = Path(saved_model_file).name
        matches = re.search(file_pattern, file_name)
        if matches:
            epoch = matches.group(1)
            logger.info(f"[{idx}/{len(saved_model_files)-1}] Processing [{saved_model_file}] -> epoch [{epoch}].")

            model = load_model(saved_model_file)
            train_auprc_val, _ = evaluate(model, train_loader, len(train_y))
            val_auprc_val, _ = evaluate(model, val_loader, len(val_y))
            test_auprc_val, _ = evaluate(model, test_loader, len(test_y))
            
            result.append((epoch, train_auprc_val, val_auprc_val, test_auprc_val))
        else:
            raise Exception("[{saved_model_file}]: wrong file name.")

    # Sort based on epoch index.
    result.sort(key=lambda x: x[0])

    with open(f'evaluate_all_epochs.pkl', 'wb') as output:
        pickle.dump(result, output, protocol=pickle.HIGHEST_PROTOCOL)


def draw_auprc_graph():
    """Draw AUPRC Graph for training, validation and testing data, along epochs.
    """
    with open('evaluate_all_epochs.pkl', 'rb') as input:
        auprc_all_epochs = pickle.load(input)

    plt.clf()

    epochs, train_auprc_val_lst, val_auprc_val_lst, test_auprc_val_lst  = list(zip(*auprc_all_epochs))

    plt.figure(figsize=(1600/230, 800/230), dpi=230)

    plt.plot(epochs, train_auprc_val_lst, label='Training AUPRC', color="blue")
    plt.plot(epochs, val_auprc_val_lst, label='Validation AUPRC', color="orange")
    plt.plot(epochs, test_auprc_val_lst, label='Test AUPRC', color="green")

    plt.axvline(x=31, color='blue', linestyle='--', linewidth=1)
    plt.axvline(x=11.95, color='orange', linestyle='--', linewidth=1)
    plt.axvline(x=12.05, color='green', linestyle='--', linewidth=1)

    plt.title('Training, Validation & Testing AUPRC')
    plt.xlabel('Epochs')
    plt.ylabel('AUPRC')

    plt.text(12+ 0.5, auprc_all_epochs[12][1]-0.025, f'{round(auprc_all_epochs[12][1], 6)}', color='blue')
    plt.text(12+ 0.5, auprc_all_epochs[12][2]-0.05, f'{round(auprc_all_epochs[12][2], 6)}', color='orange')
    plt.text(12+ 0.5, auprc_all_epochs[12][3]+0.020, f'{round(auprc_all_epochs[12][3], 6)}', color='green')

    plt.text(31+ 0.17, auprc_all_epochs[31][1]-0.025, f'{round(auprc_all_epochs[31][1], 6)}', color='blue')

    plt.xticks(epochs, fontsize=7) 
    plt.legend(loc='best')

    plt.savefig('visual.png', bbox_inches='tight')


if __name__ == '__main__':
    logger.info("*** BEGIN ***")
    # Note: Typically, collect_auprc_data() runs on Server with strong GPU. Then after the AUPRC data are saved, one can process it anywhere and draw the graph.
    # collect_auprc_data()
    draw_auprc_graph()
    logger.info("*** DONE ***")
