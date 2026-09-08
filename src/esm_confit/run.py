import sys
sys.path.append('..')

import numpy as np
import torch
from torch.amp import autocast, GradScaler
import esm
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import *
from src.esm_confit.data import *
from src.esm_confit.loss import *
from src.constants import *
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef

def finetune_confit_optimized(training_dataset, validation_dataset, model, protein, batch_converter,
                    max_epoch, batch_size, learning_rate, lambda_reg, early_stop, device="cpu",temperature=1.0, 
                    accumulation_steps = 1.0, use_mixed_precision = True):
    
    """
    Optimized training with precomputation, mixed precision, and gradient accumulation
    """
    if protein == 'WT':
        seq = LACI_WT
    elif protein == 'IA5':
        seq = IA5_WT
    elif protein == 'IA9':
        seq = IA9_WT
    else:
        raise ValueError("protein not recognized")
    
    print("Precomputing wild-type encodings and regularization logits...")
    
    # ========== PRECOMPUTATION PHASE ==========
    
    # 1. Precompute wild-type encoding ONCE
    _, _, wt_tokens = batch_converter(((protein, seq),))
    wt_tokens = wt_tokens.to(device)
    print(f"Wild-type tokens shape: {wt_tokens.shape}")
    
    # 2. Setup regularization model
    model_reg = copy.deepcopy(model)
    model_reg.to(device)
    model_reg.eval()

    # 3. Precompute regularization logits ONCE
    with torch.no_grad():
        if use_mixed_precision:
            with autocast(device):
                output_reg = model_reg(wt_tokens, repr_layers=[model_reg.num_layers], return_contacts=False)
                logits_reg_single = output_reg["logits"].float()  # Keep in FP32 for stability
        else:
            output_reg = model_reg(wt_tokens, repr_layers=[model_reg.num_layers], return_contacts=False)
            logits_reg_single = output_reg["logits"]
    
    print(f"Precomputed regularization logits shape: {logits_reg_single.shape}")
    print("Precomputation complete!")

    # Clear some memory
    del model_reg, output_reg
    torch.cuda.empty_cache()
    
    # ========== TRAINING SETUP ==========

    model.to(device)
    effective_batch_size = batch_size * accumulation_steps

    # DataLoaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_sequence_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_sequence_dataset, drop_last=True)
    
    # Optimizer and scheduler
    if isinstance(learning_rate, list):
        # learning rate schedule should be provided as [peak_learning_rate, weight_decay, warmup_step_frac]
        print("using cosine annealing with warmup LR scheduler")
        total_steps = max_epoch*len(training_dataloader) // accumulation_steps
        warmup_steps = int(learning_rate[2]*total_steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate[0], weight_decay=learning_rate[1])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scheduling = True
        print(f"Total optimization steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
    else:
        # use fixed learning rate
        print("using a fixed learning rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduling = False

    # Mixed precision setup
    if use_mixed_precision:
        scaler = GradScaler(device)
        print("Using mixed precision training")
    else:
        scaler = None
        print("Using FP32 training")

    # ========== TRAINING LOOP ==========

    print("training starts...")
    training_loss_record = []
    training_bt_loss_record = []
    training_kl_loss_record = []
    validation_loss_record = []
    validation_bt_loss_record = []
    validation_kl_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    # Initial validation
    val_loss, val_bt_loss, val_kl_loss, val_metric = run_validation_optimized(
        model, validation_dataloader, wt_tokens, logits_reg_single, lambda_reg, 
        batch_converter, device, use_mixed_precision
    )
    validation_loss_record.append(val_loss)
    validation_bt_loss_record.append(val_bt_loss)
    validation_kl_loss_record.append(val_kl_loss)
    validation_metric_record.append(val_metric)
    print(f"Initial validation - Loss: {val_loss:.4f}, BT: {val_bt_loss:.4f}, KL: {val_kl_loss:.4f}, Metric: {val_metric:.4f}")

    for e in range(max_epoch):

        if early_stop != None:
            if (e+1-best_epoch) > early_stop:
                # if the loss did not decrease for pre-defined number of epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
         # Training phase with gradient accumulation
        train_loss, train_bt_loss, train_kl_loss = run_training_optimized(
            model, training_dataloader, wt_tokens, logits_reg_single, optimizer, scheduler,
            scaler, lambda_reg, accumulation_steps, batch_converter, device, 
            use_mixed_precision, scheduling
        )
        
        training_loss_record.append(train_loss)
        training_bt_loss_record.append(train_bt_loss)
        training_kl_loss_record.append(train_kl_loss)

        # Validation phase
        val_loss, val_bt_loss, val_kl_loss, val_metric = run_validation_optimized(
            model, validation_dataloader, wt_tokens, logits_reg_single, lambda_reg, 
            batch_converter, device, use_mixed_precision
        )
        
        validation_loss_record.append(val_loss)
        validation_bt_loss_record.append(val_bt_loss)
        validation_kl_loss_record.append(val_kl_loss)
        validation_metric_record.append(val_metric)
        
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = e + 1
            best_model = model.state_dict()

        print(f"Train Loss: {train_loss:.4f} (BT: {train_bt_loss:.4f}, KL: {train_kl_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (BT: {val_bt_loss:.4f}, KL: {val_kl_loss:.4f})")
        print(f"Val Spearman: {val_metric:.4f}")

    print(f"Best model at epoch {best_epoch}")

    result_dict = {
        "training_epoch": e + 1, 
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "training_loss_record": training_loss_record,
        "validation_loss_record": validation_loss_record,
        "training_bt_loss_record": training_bt_loss_record,
        "training_kl_loss_record": training_kl_loss_record,
        "validation_bt_loss_record": validation_bt_loss_record, 
        "validation_kl_loss_record": validation_kl_loss_record,
        "validation_metric_record": validation_metric_record
    }

    model.load_state_dict(best_model)

    return model, result_dict

def run_training_optimized(model, dataloader, wt_tokens, logits_reg_single, optimizer, scheduler,
                          scaler, lambda_reg, accumulation_steps, batch_converter, device, 
                          use_mixed_precision, scheduling):
    """
    Optimized training loop with precomputed values
    """
    
    model.train()
    total_loss = 0.0
    total_bt_loss = 0.0
    total_kl_loss = 0.0
    step_count = 0
    accumulated_loss = 0.0
    accumulated_bt_loss = 0.0
    accumulated_kl_loss = 0.0
    
    for batch_idx, (mutation_sequence_pair_batch, label_batch) in enumerate(tqdm(dataloader)):
        
        current_batch_size = len(label_batch)
        
        # Prepare data
        mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Expand precomputed wild-type tokens to match current batch size
        wt_batch = wt_tokens.repeat(current_batch_size, 1)
        
        # Expand precomputed regularization logits
        logits_reg = logits_reg_single.repeat(current_batch_size, 1, 1)
        
        # Forward pass with optional mixed precision
        if use_mixed_precision:
            with autocast(device):
                mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch, batch_converter)
                loss_BT = compute_BT_loss(mutation_scores, label_batch)
                loss_KL = compute_KL_loss(logits, logits_reg, wt_batch)
                loss_batch = (loss_BT + lambda_reg * loss_KL) / accumulation_steps
        else:
            mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch, batch_converter)
            loss_BT = compute_BT_loss(mutation_scores, label_batch)
            loss_KL = compute_KL_loss(logits, logits_reg, wt_batch)
            loss_batch = (loss_BT + lambda_reg * loss_KL) / accumulation_steps
        
        # Backward pass
        if use_mixed_precision:
            scaler.scale(loss_batch).backward()
        else:
            loss_batch.backward()
        
        # Accumulate loss values for logging
        accumulated_loss += loss_batch.item()
        accumulated_bt_loss += (loss_BT.item() / accumulation_steps)
        accumulated_kl_loss += (loss_KL.item() / accumulation_steps)
        
        # Update weights when accumulation is complete
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            if scheduling:
                scheduler.step()
            
            optimizer.zero_grad()
            
            # Log accumulated losses
            total_loss += accumulated_loss * accumulation_steps
            total_bt_loss += accumulated_bt_loss * accumulation_steps
            total_kl_loss += accumulated_kl_loss * accumulation_steps
            
            # Reset accumulators
            accumulated_loss = 0.0
            accumulated_bt_loss = 0.0
            accumulated_kl_loss = 0.0
            step_count += 1
    
    # Return average losses
    avg_loss = total_loss / step_count if step_count > 0 else 0.0
    avg_bt_loss = total_bt_loss / step_count if step_count > 0 else 0.0
    avg_kl_loss = total_kl_loss / step_count if step_count > 0 else 0.0
    
    return avg_loss, avg_bt_loss, avg_kl_loss


def run_validation_optimized(model, dataloader, wt_tokens, logits_reg_single, lambda_reg, 
                            batch_converter, device, use_mixed_precision):
    """
    Optimized validation loop with precomputed values
    """
    from torchmetrics.regression import SpearmanCorrCoef
    
    model.eval()
    total_loss = 0.0
    total_bt_loss = 0.0
    total_kl_loss = 0.0
    batch_count = 0
    
    CRITERION = SpearmanCorrCoef()
    CRITERION.reset()
    
    with torch.no_grad():
        for mutation_sequence_pair_batch, label_batch in tqdm(dataloader):
            
            current_batch_size = len(label_batch)
            
            # Prepare data
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            
            # Expand precomputed values
            wt_batch = wt_tokens.repeat(current_batch_size, 1)
            logits_reg = logits_reg_single.repeat(current_batch_size, 1, 1)
            
            # Forward pass
            if use_mixed_precision:
                with autocast(device):
                    mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch, batch_converter)
                    loss_BT = compute_BT_loss(mutation_scores, label_batch)
                    loss_KL = compute_KL_loss(logits, logits_reg, wt_batch)
                    loss_batch = loss_BT + lambda_reg * loss_KL
            else:
                mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch, batch_converter)
                loss_BT = compute_BT_loss(mutation_scores, label_batch)
                loss_KL = compute_KL_loss(logits, logits_reg, wt_batch)
                loss_batch = loss_BT + lambda_reg * loss_KL
            
            total_loss += loss_batch.item()
            total_bt_loss += loss_BT.item()
            total_kl_loss += loss_KL.item()
            batch_count += 1
            
            CRITERION.update(mutation_scores, label_batch)
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_bt_loss = total_bt_loss / batch_count if batch_count > 0 else 0.0
    avg_kl_loss = total_kl_loss / batch_count if batch_count > 0 else 0.0
    metric = CRITERION.compute().item()
    
    return avg_loss, avg_bt_loss, avg_kl_loss, metric

def evaluate_confit(testing_dataset, model, batch_converter, protein, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False, 
                                    collate_fn=collate_function_sequence_dataset, drop_last=False)
    
    model.to(device)

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

    # Precompute wild-type encoding ONCE
    _, _, wt_tokens = batch_converter(((protein, seq),))
    wt_tokens = wt_tokens.to(device)

    model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(testing_dataloader):

        with torch.no_grad():
            current_batch_size = len(label_batch)
            
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            
            # Expand precomputed wild-type tokens to match current batch size
            wt_batch = wt_tokens.repeat(current_batch_size, 1)

            mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch, batch_converter)
            CRITERION.update(mutation_scores, label_batch)

    return CRITERION.compute().item()


def nfold_finetune_confit(data_dir, saving_dir, file_name, pretrained_model, protein, nfold, max_epoch, batch_size, 
                     learning_rate, lambda_reg, early_stop, device, temperature,accumulation_steps,use_mixed_precision):

    nfold_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]
        testing_data = data_nfold[testing_idx]

        training_dataset = SequenceDataset(training_data)
        validation_dataset = SequenceDataset(validation_data)
        testing_dataset = SequenceDataset(testing_data)
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
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        best_model, result_dict = finetune_confit_optimized(training_dataset, validation_dataset, ESM2_model, protein, batch_converter,
                                                    max_epoch, batch_size, learning_rate, lambda_reg, 
                                                    early_stop, device, temperature, accumulation_steps, use_mixed_precision)

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        torch.save(best_model.state_dict(), saving_dir+"ESM2_confit_fold{}_model.pt".format(n))

    save_data(saving_dir+"ESM2_confit_result_dict.pkl", nfold_result_dict)


def nfold_evaluate_confit_single_mutation(saved_dir, data_dir, file_name, pretrained_model, nfold, protein, task, device):
    """
    evaluate n-fold saved confit models on the given task
    """

    test_metric_list = []
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        testing_data = data_nfold[testing_idx]
        testing_dataset = SequenceDataset(testing_data)
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
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        ESM2_model.load_state_dict(torch.load(saved_dir + "ESM2_confit_fold{}_model.pt".format(n)))

        test_metric = evaluate_confit(testing_dataset, ESM2_model, batch_converter, protein, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("\n\navg binary AUC: {}".format(np.mean(test_metric_list)))
        print("\n\nstd binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("\n\navg spearman corr: {}".format(np.mean(test_metric_list)))
        print("\n\nstd spearman corr: {}".format(np.std(test_metric_list)))

    print(test_metric_list)
    
    return test_metric_list


def nfold_evaluate_confit_multiple_mutations(saved_dir, data_dir, pretrained_model, nfold, task, device):
    """
    evaluate n-fold saved confit models on the given task
    """

    test_metric_list = []
    testing_data = load_data(data_dir)
    testing_dataset = SequenceDataset(testing_data)

    for n in range(nfold):

        print("loading pre-trained ESM2...")
        if pretrained_model == "esm2_t12_35M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t30_150M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t33_650M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        ESM2_model.load_state_dict(torch.load(saved_dir + "ESM2_confit_fold{}_model.pt".format(n)))

        test_metric = evaluate_confit(testing_dataset, ESM2_model, batch_converter, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))

    print(test_metric_list)
    
    return test_metric_list