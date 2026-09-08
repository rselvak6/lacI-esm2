import sys
sys.path.append('..')

from src.head_finetune.data import *
from src.head_finetune.models import *
from src.head_finetune.utils import *
from src.utils import *
from tqdm import tqdm
import esm
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef


def train_meanpooling_head(training_dataset, validation_dataset, input_dim, max_epoch, batch_size, learning_rate, 
                           early_stop=5, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_representation_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_representation_dataset, drop_last=True)
 
    model = MeanPoolingHead(input_dim, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    model.eval()
    for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            loss_batch = loss_fn(logits, label_batch)
            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    validation_loss_record.append(validation_loss)
    print("initial validation loss: {}".format(validation_loss))

    for e in range(max_epoch):

        if early_stop != None:
            if (e+1-best_epoch) > early_stop:
                # if the loss did not decrease for pre-defined number of epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        model.train()
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            loss_batch = loss_fn(logits, label_batch)
            loss_batch.backward()
            Optimizer.step()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)

        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                representation_batch = representation_batch.to(device)
                label_batch = label_batch.to(device)
                logits = model(representation_batch) # (batch_size, 1)
                label_batch = label_batch.view(logits.size()) # (batch_size, 1)
                loss_batch = loss_fn(logits, label_batch)
                CRITERION.update(logits, label_batch)
                batch_loss_sum += loss_batch.item()
        
        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_record.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))
        print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate,
                   "training_loss_record" : training_loss_record, 
                   "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict

def train_attention_head(training_dataset, validation_dataset, input_dim, max_epoch, batch_size, learning_rate,
                         early_stop=5, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_representation_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_representation_dataset, drop_last=True)
 
    model = AttentionHead(input_dim, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    model.eval()
    for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            loss_batch = loss_fn(logits, label_batch)
            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    validation_loss_record.append(validation_loss)
    print("initial validation loss: {}".format(validation_loss))

    for e in range(max_epoch):

        if early_stop != None:
            if (e+1-best_epoch) > early_stop:
                # if the loss did not decrease for pre-defined number of epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        model.train()
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            loss_batch = loss_fn(logits, label_batch)
            loss_batch.backward()
            Optimizer.step()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                representation_batch = representation_batch.to(device)
                label_batch = label_batch.to(device)
                logits = model(representation_batch) # (batch_size, 1)
                label_batch = label_batch.view(logits.size()) # (batch_size, 1)
                loss_batch = loss_fn(logits, label_batch)
                CRITERION.update(logits, label_batch)
                batch_loss_sum += loss_batch.item()
        
        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_record.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))
        print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate,
                   "training_loss_record" : training_loss_record, 
                   "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict

def train_position_head(training_dataset, validation_dataset, input_dim, max_epoch, batch_size, learning_rate, 
                        early_stop=5, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_representation_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_representation_dataset, drop_last=True)
 
    model = MutationPositionHead(input_dim, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    model.eval()
    for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch, position) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            loss_batch = loss_fn(logits, label_batch)
            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    validation_loss_record.append(validation_loss)
    print("initial validation loss: {}".format(validation_loss))

    for e in range(max_epoch):

        if early_stop != None:
            if (e+1-best_epoch) > early_stop:
                # if the loss did not decrease for pre-defined number of epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        model.train()
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch, position) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            loss_batch = loss_fn(logits, label_batch)

            loss_batch.backward()
            Optimizer.step()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                representation_batch = representation_batch.to(device)
                label_batch = label_batch.to(device)
                logits = model(representation_batch, position) # (batch_size, 1)
                label_batch = label_batch.view(logits.size()) # (batch_size, 1)
                loss_batch = loss_fn(logits, label_batch)
                CRITERION.update(logits, label_batch)
                batch_loss_sum += loss_batch.item()
        
        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_record.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))
        print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate,
                   "training_loss_record" : training_loss_record, 
                   "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict


def evaluate_head_single_mutation(testing_dataset, model, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, 
                                    collate_fn=collate_function_representation_dataset, drop_last=True)

    if task == "binary":
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    model.to(device)
    model.eval()
    for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(testing_dataloader):

        with torch.no_grad():
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            if model.__class__.__name__ == "MutationPositionHead":
                logits = model(representation_batch, position.view(representation_batch.size(0),-1)) # (batch_size, 1)
            else:
                logits = model(representation_batch) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            CRITERION.update(logits, label_batch)

    return CRITERION.compute().item()


def nfold_train_head(mutation_data_dir, representation_data_dir, saving_dir, mutation_file_name, 
                    pretrained_model, model_name, nfold, max_epoch, batch_size, learning_rate, early_stop, device):

    nfold_result_dict = dict()
    representation_data_dict = load_data(representation_data_dir + pretrained_model + "_single_mutation_representaton_data_dict.pkl")
    data_nfold = np.array(load_nfold_data(mutation_data_dir, mutation_file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    if pretrained_model == "esm2_t6_8M_UR50D":
        embed_dim = 320
    elif pretrained_model == "esm2_t12_35M_UR50D":
        embed_dim = 480
    elif pretrained_model == "esm2_t33_650M_UR50D":
        embed_dim = 1280
    elif pretrained_model == "esm2_t36_3B_UR50D":
        embed_dim = 2560

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]

        training_dataset = RepresentatonDataset(training_data, representation_data_dict)
        validation_dataset = RepresentatonDataset(validation_data, representation_data_dict) 
        print("loading data done")

        if model_name == "attention":
            best_model, result_dict = train_attention_head(training_dataset, validation_dataset, embed_dim, max_epoch, 
                                                        batch_size, learning_rate, early_stop, device)
        elif model_name == "mean":
            best_model, result_dict = train_meanpooling_head(training_dataset, validation_dataset, embed_dim, max_epoch, 
                                                        batch_size, learning_rate, early_stop, device)
        elif model_name == "position":
            best_model, result_dict = train_position_head(training_dataset, validation_dataset, embed_dim, max_epoch, 
                                                        batch_size, learning_rate, early_stop, device)
        else:
            raise ValueError("wrong model head input")

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        torch.save(best_model.state_dict(), saving_dir + model_name + "_fold{}_model.pt".format(n))

    save_data(saving_dir + model_name + "_result_dict.pkl", nfold_result_dict)


def nfold_evaluate_head_single_mutation(mutation_data_dir, representation_data_dir, saved_dir, mutation_file_name,
                                        pretrained_model, model_name, nfold, task, device):
    """
    evaluate n-fold saved models on the given task
    """

    test_metric_list = []
    representation_data_dict = load_data(representation_data_dir + pretrained_model + "_single_mutation_representaton_data_dict.pkl")
    data_nfold = np.array(load_nfold_data(mutation_data_dir, mutation_file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    if pretrained_model == "esm2_t6_8M_UR50D":
        embed_dim = 320
    elif pretrained_model == "esm2_t12_35M_UR50D":
        embed_dim = 480
    elif pretrained_model == "esm2_t33_650M_UR50D":
        embed_dim = 1280
    elif pretrained_model == "esm2_t36_3B_UR50D":
        embed_dim = 2560

    for n in range(nfold):

        testing_idx = np.array(nfold_idx[n+1])
        testing_data = data_nfold[testing_idx]
        testing_dataset = RepresentatonDataset(testing_data, representation_data_dict)

        if model_name == "attention":
            model = AttentionHead(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + model_name + "_fold{}_model.pt".format(n)))
        elif model_name == "mean":
            model = MeanPoolingHead(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + model_name + "_fold{}_model.pt".format(n)))
        elif model_name == "position":
            model = MutationPositionHead(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + model_name + "_fold{}_model.pt".format(n)))

        test_metric = evaluate_head_single_mutation(testing_dataset, model, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("\n\navg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("\n\navg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
        
    return test_metric_list


def nfold_evaluate_head_multiple_mutations(mutation_data_dir, saved_dir,
                                           pretrained_model, model_name, nfold, score_computation, task, device):
    """
    evaluate n-fold saved models on the given task
    input representations of proteins are generated using ESM model, not from saved dataset 
    this is because it is not possible to save all possible double mutations 
    """

    test_metric_list = []
    testing_data = load_data(mutation_data_dir)
    testing_dataset = SequenceDataset(testing_data)

    print("loading pre-trained ESM2...")
    if pretrained_model == "esm2_t12_35M_UR50D":
        ESM2_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        embed_dim = ESM2_model.embed_dim
    elif pretrained_model == "esm2_t30_150M_UR50D":
        ESM2_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        embed_dim = ESM2_model.embed_dim
    elif pretrained_model == "esm2_t33_650M_UR50D":
        ESM2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        embed_dim = ESM2_model.embed_dim
    else:
        raise ValueError("wrong pretrained model input")
    print("loading pre-trained ESM2 done")

    for n in range(nfold):

        if model_name == "attention":
            model = AttentionHead(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + model_name + "_fold{}_model.pt".format(n)))
        elif model_name == "mean":
            model = MeanPoolingHead(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + model_name + "_fold{}_model.pt".format(n)))
        elif model_name == "position":
            model = MutationPositionHead(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + model_name + "_fold{}_model.pt".format(n)))

        test_metric = evaluate_head_multiple_mutations(testing_dataset, ESM2_model, model, model_name, batch_converter,
                                                       score_computation, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
        
    return test_metric_list


def evaluate_head_multiple_mutations(testing_dataset, pretrained_ESM, model, model_name, batch_converter, 
                                     score_computation, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False, 
                                    collate_fn=collate_function_sequence_dataset, drop_last=False)
    
    pretrained_ESM.to(device)
    model.to(device)

    if task == "binary":
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    model.eval()
    pretrained_ESM.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(testing_dataloader):

        with torch.no_grad():
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)

            if model_name in ["mean", "attention"]:
                if score_computation == "general":
                    mutation_scores = compute_general_mutation_score(model, pretrained_ESM, x_batch)
                elif score_computation == "additive":
                    mutation_scores = compute_additive_mutation_score(model, pretrained_ESM, 
                                                                      x_batch, mutation_batch, 
                                                                      batch_converter)
            if model_name == "position":
                if score_computation == "general":
                    mutation_scores = compute_general_mutation_score_position_pooling(model, pretrained_ESM, 
                                                                                      x_batch, mutation_batch)
                elif score_computation == "additive":
                    mutation_scores = compute_additive_mutation_score_position_pooling(model, 
                                                                                       x_batch, 
                                                                                       mutation_batch, 
                                                                                       batch_converter)

        CRITERION.update(mutation_scores.to(device), label_batch.view(mutation_scores.size()))

    return CRITERION.compute().item()