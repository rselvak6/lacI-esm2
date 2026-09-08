import numpy as np
import torch
from torch.amp import autocast, GradScaler
import esm
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import *
from src.full_finetune.data import *
from src.full_finetune.models import *
from src.full_finetune.utils import *
from src.constants import *
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef

def finetune_ESM2_position_head_optimized(training_dataset, validation_dataset, pretrained_model, batch_converter,
                                         max_epoch, batch_size, learning_rate, early_stop=5, device="cpu",
                                         accumulation_steps=1, use_mixed_precision=True):
    """
    Optimized full fine-tuning with mixed precision and gradient accumulation
    """
    
    # Calculate effective batch size
    effective_batch_size = batch_size * accumulation_steps
    print(f"Physical batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    # DataLoaders - use drop_last=False to handle variable batch sizes
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_sequence_dataset, drop_last=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_sequence_dataset, drop_last=False)
 
    # Model setup
    model = ESM2MutationPositionHead(pretrained_model, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()

    # Optimizer and scheduler setup
    if isinstance(learning_rate, list):
        print("Using cosine annealing with warmup LR scheduler")
        # Scale total steps by accumulation steps since we update less frequently
        total_steps = max_epoch * len(training_dataloader) // accumulation_steps
        warmup_steps = int(learning_rate[2] * total_steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate[0], weight_decay=learning_rate[1])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scheduling = True
        print(f"Total optimization steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
    else:
        print("Using a fixed learning rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduling = False

    # Mixed precision setup
    if use_mixed_precision:
        scaler = GradScaler(device)
        print("Using mixed precision training")
    else:
        scaler = None
        print("Using FP32 training")

    print("Training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    # Initial validation
    initial_val_loss, initial_val_metric = run_validation_optimized(
        model, validation_dataloader, batch_converter, loss_fn, device, use_mixed_precision
    )
    validation_loss_record.append(initial_val_loss)
    validation_metric_record.append(initial_val_metric)
    print(f"Initial validation loss: {initial_val_loss:.4f}")
    print(f"Initial validation Spearman: {initial_val_metric:.4f}")

    for e in range(max_epoch):
        
        if early_stop is not None:
            if (e + 1 - best_epoch) > early_stop:
                break

        print(f"Epoch {e+1}")
        
        # Training phase with gradient accumulation
        training_loss = run_training_optimized(
            model, training_dataloader, batch_converter, loss_fn, optimizer, scheduler,
            scaler, accumulation_steps, device, use_mixed_precision, scheduling
        )
        training_loss_record.append(training_loss)

        # Validation phase
        validation_loss, validation_metric = run_validation_optimized(
            model, validation_dataloader, batch_converter, loss_fn, device, use_mixed_precision
        )
        validation_loss_record.append(validation_loss)
        validation_metric_record.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e + 1
            best_model = model.state_dict()

        print(f"Training loss: {training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation Spearman corr: {validation_metric:.4f}")

    print(f"Best model at epoch {best_epoch}")
    print("Saving results...")
    
    result_dict = {
        "training_epoch": e + 1, 
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "training_loss_record": training_loss_record, 
        "validation_loss_record": validation_loss_record,
        "validation_metric_record": validation_metric_record
    }

    model.load_state_dict(best_model)
    return model, result_dict


def run_training_optimized(model, dataloader, batch_converter, loss_fn, optimizer, scheduler,
                          scaler, accumulation_steps, device, use_mixed_precision, scheduling):
    """
    Optimized training loop with gradient accumulation and mixed precision
    """
    
    model.train()
    total_loss = 0.0
    step_count = 0
    accumulated_loss = 0.0
    
    for batch_idx, (mutation_sequence_pair_batch, label_batch) in enumerate(tqdm(dataloader)):
        
        current_batch_size = len(label_batch)
        
        # Prepare data
        mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
        mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
        mutation_position_batch = mutation_position_batch.to(device)
        label_batch = label_batch.to(dtype=torch.float32).to(device)
        x_batch = x_batch.to(device)
        
        # Forward pass with optional mixed precision
        if use_mixed_precision:
            with autocast(device):
                logits = model(x_batch, mutation_position_batch)
                label_batch_reshaped = label_batch.view(logits.size())
                loss_batch = loss_fn(logits, label_batch_reshaped) / accumulation_steps
        else:
            logits = model(x_batch, mutation_position_batch)
            label_batch_reshaped = label_batch.view(logits.size())
            loss_batch = loss_fn(logits, label_batch_reshaped) / accumulation_steps
        
        # Backward pass
        if use_mixed_precision:
            scaler.scale(loss_batch).backward()
        else:
            loss_batch.backward()
        
        # Accumulate loss for logging
        accumulated_loss += loss_batch.item()
        
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
            
            # Log accumulated loss
            total_loss += accumulated_loss * accumulation_steps
            accumulated_loss = 0.0
            step_count += 1
    
    # Return average loss
    avg_loss = total_loss / step_count if step_count > 0 else 0.0
    return avg_loss


def run_validation_optimized(model, dataloader, batch_converter, loss_fn, device, use_mixed_precision):
    """
    Optimized validation loop with mixed precision
    """
    
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    CRITERION = SpearmanCorrCoef()
    CRITERION.reset()
    
    with torch.no_grad():
        for mutation_sequence_pair_batch, label_batch in tqdm(dataloader):
            
            # Prepare data
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
            mutation_position_batch = mutation_position_batch.to(device)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)
            
            # Forward pass
            if use_mixed_precision:
                with autocast(device):
                    logits = model(x_batch, mutation_position_batch)
                    label_batch_reshaped = label_batch.view(logits.size())
                    loss_batch = loss_fn(logits, label_batch_reshaped)
            else:
                logits = model(x_batch, mutation_position_batch)
                label_batch_reshaped = label_batch.view(logits.size())
                loss_batch = loss_batch = loss_fn(logits, label_batch_reshaped)
            
            total_loss += loss_batch.item()
            batch_count += 1
            
            # Update metric
            CRITERION.update(logits, label_batch_reshaped)
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    metric = CRITERION.compute().item()
    
    # Reset metric to free memory
    CRITERION.reset()
    
    return avg_loss, metric


def evaluate_full_finetuned_model_single_mutation_optimized(testing_dataset, model, head, batch_converter, 
                                                           task, device, use_mixed_precision=True):
    """
    Optimized evaluation function with mixed precision
    """

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False, 
                                    collate_fn=collate_function_sequence_dataset, drop_last=False)
        
    model.to(device)

    if task == "binary":
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    model.eval()
    with torch.no_grad():
        for mutation_sequence_pair_batch, label_batch in tqdm(testing_dataloader):

            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)

            # Forward pass with optional mixed precision
            if use_mixed_precision:
                with autocast(device):
                    if head in ["mean", "attention"]:
                        mutation_scores = model(x_batch) 
                    elif head == "position":
                        mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
                        mutation_position_batch = mutation_position_batch.to(device)
                        mutation_scores = model(x_batch, mutation_position_batch)
            else:
                if head in ["mean", "attention"]:
                    mutation_scores = model(x_batch) 
                elif head == "position":
                    mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
                    mutation_position_batch = mutation_position_batch.to(device)
                    mutation_scores = model(x_batch, mutation_position_batch)

            CRITERION.update(mutation_scores.to(device), label_batch.view(mutation_scores.size()))

    result = CRITERION.compute().item()
    CRITERION.reset()  # Free memory
    return result


def nfold_full_finetune_optimized(data_dir, saving_dir, file_name, pretrained_model, head, nfold, 
                                 max_epoch, batch_size, learning_rate, early_stop, device,
                                 accumulation_steps=1, use_mixed_precision=True):
    """
    Optimized n-fold full fine-tuning with all performance improvements
    """

    nfold_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold))
    nfold_idx = np.tile(np.arange(nfold), 2)

    print(f"Starting n-fold training with optimizations:")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Mixed precision: {use_mixed_precision}")

    for n in range(nfold):
        
        print(f"\n=== Fold {n+1}/{nfold} ===")
        print("Loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]

        training_dataset = SequenceDataset(training_data)
        validation_dataset = SequenceDataset(validation_data)
        print("Loading data done")

        print("Loading pre-trained ESM2...")
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
            raise ValueError("Wrong pretrained model input")
        print("Loading pre-trained ESM2 done")

        # Enable gradient checkpointing for memory efficiency (especially for larger models)
        if hasattr(ESM2_model, 'gradient_checkpointing_enable'):
            ESM2_model.gradient_checkpointing_enable()

        # Train with optimizations
        if head == "attention":
            best_model, result_dict = finetune_ESM2_attention_head_optimized(
                training_dataset, validation_dataset, ESM2_model, batch_converter, 
                max_epoch, batch_size, learning_rate, early_stop, device,
                accumulation_steps, use_mixed_precision
            )
        elif head == "mean":
            best_model, result_dict = finetune_ESM2_meanpooling_head_optimized(
                training_dataset, validation_dataset, ESM2_model, batch_converter,
                max_epoch, batch_size, learning_rate, early_stop, device,
                accumulation_steps, use_mixed_precision
            )
        elif head == "position":
            best_model, result_dict = finetune_ESM2_position_head_optimized(
                training_dataset, validation_dataset, ESM2_model, batch_converter,
                max_epoch, batch_size, learning_rate, early_stop, device,
                accumulation_steps, use_mixed_precision
            )   
        else:
            raise ValueError("Wrong head value assigned")

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        torch.save(best_model.state_dict(), saving_dir + head + "_fold{}_model.pt".format(n))

        # Clean up GPU memory between folds
        del ESM2_model, best_model
        torch.cuda.empty_cache()

    save_data(saving_dir + head + "_result_dict.pkl", nfold_result_dict)
    print(f"\nN-fold training complete! Results saved to {saving_dir}")


def nfold_evaluate_full_finetune_single_mutation_optimized(saved_dir, data_dir, file_name, pretrained_model, 
                                                          head, nfold, task, device, use_mixed_precision=True):
    """
    Optimized n-fold evaluation with mixed precision
    """

    test_metric_list = []
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold))
    nfold_idx = np.tile(np.arange(nfold), 2)

    print(f"Starting n-fold evaluation with mixed precision: {use_mixed_precision}")

    for n in range(nfold):
        
        print(f"\n=== Evaluating Fold {n+1}/{nfold} ===")
        print("Loading data...")
        testing_idx = np.array(nfold_idx[n+1])
        testing_data = data_nfold[testing_idx]
        testing_dataset = SequenceDataset(testing_data)
        print("Loading data done")

        print("Loading pre-trained ESM2...")
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
            raise ValueError("Wrong pretrained model input")
        print("Loading pre-trained ESM2 done")

        # Load the appropriate model
        if head == "attention":
            model = ESM2AttentionHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))
        elif head == "mean":
            model = ESM2MeanpoolingHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))
        elif head == "position":
            model = ESM2MutationPositionHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))

        # Evaluate with optimizations
        test_metric = evaluate_full_finetuned_model_single_mutation_optimized(
            testing_dataset, model, head, batch_converter, task, device, use_mixed_precision
        )
        test_metric_list.append(test_metric)

        # Clean up memory
        del ESM2_model, model
        torch.cuda.empty_cache()

    # Print results
    if task == "binary":
        print(f"\n\nAvg binary AUC: {np.mean(test_metric_list):.4f}")
        print(f"Std binary AUC: {np.std(test_metric_list):.4f}")
    elif task == "score":
        print(f"\n\nAvg Spearman corr: {np.mean(test_metric_list):.4f}")
        print(f"Std Spearman corr: {np.std(test_metric_list):.4f}")

    print(f"Individual fold results: {test_metric_list}")
    return test_metric_list