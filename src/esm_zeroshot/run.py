import torch
import esm
from tqdm import tqdm

import sys
sys.path.append('..')

from src.utils import *
from data import *
from utils import *
from src.constants import *
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef


def evaluate_esm_zeroshot(mutation_data, model, batch_converter, task, protein, random_weight=False, device="cpu"):
    
    if random_weight:
        # if random_weight=True, randomly initialized the esm model weights
        model = init_esm_weight(model)

    if protein == 'WT':
        seq = LACI_WT
    elif protein == 'IA5':
        seq = IA5_WT
    elif protein == 'IA9':
        seq = IA9_WT
    else:
        raise ValueError("protein not recognized")

    if task == "binary":
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
    
    data = [(protein, seq),]
    # batch converter adds <CLS> token at the first position
    # and <EOS> token at the last position
    mutation_batch, sequence_batch, x_batch = batch_converter(data)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_batch = x_batch.to(device)
        outputs = model(x_batch, repr_layers=[model.num_layers-1], return_contacts=True)
        token_probs = torch.log_softmax(outputs["logits"], dim=-1)
        
    for mutation_sequence, (mutation, label) in tqdm(mutation_data.items()):

        mutation_score = compute_zeroshot_mutation_score(mutation, seq, token_probs, batch_converter.alphabet)
        CRITERION.update(torch.tensor([mutation_score]), torch.tensor([label]).to(torch.float32))
    
    return CRITERION.compute().item()


def nfold_evaluate_zeroshot_single_mutation(data_dir, file_name, pretrained_model, nfold, task, protein, random_weight, device):
    """
    n-fold evaluation of ESM2 model on the given task using zero-shot mutation score
    """

    test_metric_list = []
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        testing_data = data_nfold[testing_idx] # zero-shot prediction does not use SequenceDataset
        print("loading data done")

        print("loading pre-trained ESM2...")
        if pretrained_model == "esm2_t6_8M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t12_35M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t30_150M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t33_650M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t36_3B_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            batch_converter = alphabet.get_batch_converter() 
        elif pretrained_model == "esm2_t48_15B_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            batch_converter = alphabet.get_batch_converter()  
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        test_metric = evaluate_esm_zeroshot(testing_data, ESM2_model, batch_converter, task, protein, random_weight, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("\navg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("\navg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
    
    return test_metric_list