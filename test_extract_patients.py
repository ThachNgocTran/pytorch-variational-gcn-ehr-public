import pickle, logging, sys
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
import torch

from model import VariationalGNN


logging.basicConfig(format="%(asctime)s %(levelname)s: [%(name)s] [%(funcName)s]: %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def do_inference(model, data):
    model.eval()
    with torch.inference_mode():
         probability, _ = model(data)
         probability = torch.sigmoid(probability.detach()).cpu()
         
    return probability


def extract_test_patients_prediction():
    """Extract truth label and prediction for 1600 first patients in test data. Write result to file.
    """
    test_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] = pickle.load(open('data/test_csr.pkl', 'rb'))
    test_x, test_y = test_xy
    test_x = test_x.toarray()

    # TO BE CHANGED HERE:
    path = r"parameter_(epoch_[12])_(idx_[1520])_(auprc_[0.7033310548760412]).pt"

    kwargs, state = torch.load(path, weights_only=False)
    model = VariationalGNN(**kwargs).to(device)
    model.load_state_dict(state)

    total_probability = torch.empty((0, 1))

    # Check 100 batches of 16 each.
    for i in range(100):
        patient = torch.from_numpy(test_x[i*16:i*16+16]).to(device)
        probability = do_inference(model, patient)
        total_probability = torch.cat([total_probability, probability], dim=0)

    first = total_probability > 0.5
    second = torch.from_numpy(test_y[0:1600]).unsqueeze(-1) == 1

    final_result = torch.cat([first, second], dim=1)

    torch.set_printoptions(profile="full")

    with open("tensor.txt", "w", encoding="utf-8") as text_file:
        print(f"{final_result}", file=text_file)


def generate_patient_csv_for_testing():
    """For the file "tensor.txt", choose 3 patients with correct prediction, and 3 patients with wrong prediction. Write those patients as separate CSV files.
    Also extract the list of all patient features, and save as "feature.csv".
    """
    # Create the total list of condition names (diagnose code, procedure code, lab values).
    with open('./data/dx_map.p', 'rb') as input:
        dx_map = pickle.load(input)
    with open('./data/proc_map.p', 'rb') as input:
        proc_map = pickle.load(input)
    with open('./data/lab_map.p', 'rb') as input:
        lab_map = pickle.load(input)
    total_list = list(dx_map.keys()) +  list(proc_map.keys()) + list(lab_map.keys())    # len= 10591
    type_list = len(dx_map) * ["ICD9-Diag"] + len(proc_map) * ["ICD9-Proc"] + len(lab_map) * ["LabValue"]
    
    ds = pd.DataFrame({'condition': total_list, 'type': type_list})
    ds.to_csv("feature.csv", sep=',', encoding='utf-8', index=False)

    # Extract the interesting patients.
    test_xy: Tuple[csr_matrix, npt.NDArray[np.float32]] = pickle.load(open('./data/test_csr.pkl', 'rb'))
    test_x, test_y = test_xy
    test_x = test_x.toarray()
    
    # CORRECT prediction
    correct_pred_patients = [3, 23, 58]
    for i in correct_pred_patients:
        label = 'dead' if test_y[i] == 1.0 else 'alive'
        predicted = label
        file_name = f"Patient_{i}_(Label-{label})_(Predicted-{predicted}).csv"
        if len(test_x[i]) != len(total_list):
            raise Exception("Impossible")

        value_column = test_x[i]
        df = pd.DataFrame({"condition": total_list, "value": value_column, "type": type_list})
        df.to_csv(file_name, index=False)
    
    # WRONG prediction
    wrong_pred_patients = [8, 22, 131]
    for i in wrong_pred_patients:
        label = 'dead' if test_y[i] == 1.0 else 'alive'
        predicted = 'alive' if label == 'dead' else 'dead'
        file_name = f"Patient_{i}_(Label-{label})_(Predicted-{predicted}).csv"
        if len(test_x[i]) != len(total_list):
            raise Exception("Impossible")

        value_column = test_x[i]
        df = pd.DataFrame({"condition": total_list, "value": value_column, "type": type_list})
        df.to_csv(file_name, index=False)


if __name__ == '__main__':
    logger.info("*** BEGIN ***")
    # First extract test patients for manual checking, then save 6 notable patients (3 right, 3 wrong) as CSV.
    # extract_test_patients_prediction()
    generate_patient_csv_for_testing()
    logger.info("*** DONE ***")
