import torch
import numpy as np
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.input_keys = list(data_dict.keys())

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self, idx):

        mutation, label = self.data_dict[self.input_keys[idx]]

        # return (mutation, sequence) and bin_label
        return (mutation, self.input_keys[idx]), label
    

def collate_function_sequence_dataset(batch):
    
    mutation_sequence_pair, label = list(zip(*batch))
    
    return mutation_sequence_pair, torch.from_numpy(np.array(label))