import pickle
import numpy as np
import copy
import torch
import math
from torch.optim.lr_scheduler import LambdaLR

def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)


def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)


def split_nfold(saving_dir, file_name, data_dict, nfold, seed=1234):
                
    key_list = list(data_dict.keys())
    np.random.seed(seed)
    np.random.shuffle(key_list)
        
    fold_size = int(np.ceil(len(key_list)/nfold))
        
    for fold in range(nfold):
        print("splitting {}-fold".format(fold+1))
        fold_data = dict()
        fold_key_list = key_list[fold_size*fold:fold_size*(fold+1)]
        print("fold size: {}".format(len(fold_key_list)))
        
        for key in fold_key_list:
                fold_data[key] = copy.deepcopy(data_dict[key])
        
        print("saving {}-fold".format(fold+1))
        saving = saving_dir + file_name + "_{}fold.pkl".format(fold)
        save_data(saving, fold_data)


def split_nfold_sequnece_data_dict(binary_data_saving_dir, bins_data_saving_dir, 
                                   binary_file_name, bins_file_name, 
                                   binary_data_dict, bins_data_dict, nfold, seed=1234):
    """
    two data dict have different number of data
    first split binary sequnece data dict and intersection of each fold with the entire bins data dict is used as
    fold for bins data dict
    """
                
    key_list = list(binary_data_dict.keys())
    np.random.seed(seed)
    np.random.shuffle(key_list)
        
    fold_size = int(np.ceil(len(key_list)/nfold))
        
    for fold in range(nfold):
        
        print("splitting {}-fold".format(fold+1))
        binary_fold_data = dict()
        bins_fold_data = dict()
        
        fold_key_list = key_list[fold_size*fold:fold_size*(fold+1)]
        
        for key in fold_key_list:
                binary_fold_data[key] = copy.deepcopy(binary_data_dict[key])
                
                try:
                    bins_fold_data[key] = copy.deepcopy(bins_data_dict[key])
                except:
                    pass
                
        print("saving {}-fold of binary data".format(fold+1))
        print("fold size: {}".format(len(binary_fold_data)))
        saving = binary_data_saving_dir + binary_file_name + "_{}fold.pkl".format(fold)
        save_data(saving, binary_fold_data)
        print("saving {}-fold of bins data".format(fold+1))
        print("fold size: {}".format(len(bins_fold_data)))
        saving = bins_data_saving_dir + bins_file_name + "_{}fold.pkl".format(fold)
        save_data(saving, bins_fold_data)


def merge_folds(fold_list):
    
    merged_dict = dict()
    
    for fold in fold_list:
        merged_dict.update(fold)
        
    return merged_dict


def load_nfold_data(data_dir, file_name, nfold):

    nfold_list = []

    for n in range(nfold):
        print("loading {}-fold data".format(n+1))
        loading = data_dir + file_name + "_{}fold.pkl".format(n)
        nfold_list.append(load_data(loading))

    return nfold_list


def generate_lr_schedule_function(warmup_steps, decay_steps):

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warm-up phase
            return current_step / warmup_steps
        else:
            # Decay phase
            decay_progress = (current_step - warmup_steps) / decay_steps
            return 1 - decay_progress * 0.9  # Decay to 0.1 of peak_lr
        
    return lr_lambda

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Create a schedule with a learning rate that decreases following cosine function
    between the initial lr set in the optimizer to 0, after a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, -1)