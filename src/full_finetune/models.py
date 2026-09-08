import torch
import copy
import torch.nn.functional as F
from src.utils import *


class MaskedConv1d(torch.nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2) # transpose 
    

class Attention1d(torch.nn.Module):
    
    def __init__(self, in_dim):
        super().__init__()
        self.layer = MaskedConv1d(in_dim, 1, 1)

    def forward(self, x, input_mask=None):
        n, ell, _ = x.shape
        attn = self.layer(x)
        attn = attn.view(n, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(), float('-inf'))
        attn = F.softmax(attn, dim=-1).view(n, -1, 1)
        out = (attn * x).sum(dim=1)
        return out
    

class ESM2MeanpoolingHead(torch.nn.Module):
        
    def __init__(self, ESM2_model, output_dim):
        super(ESM2MeanpoolingHead, self).__init__()
        
        self.ESM2_model = copy.deepcopy(ESM2_model)
        self.embed_dim = self.ESM2_model.embed_dim
        self.ESM2_num_layers = self.ESM2_model.num_layers
        self.linear1 = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.embed_dim, output_dim)
    
    def forward(self, x):
        """
        TODO: this mean pooling does not work on sequences that have different lengths due to padding tokens
                mean pooling of sequences that have different lengths

        args
        ----
            x: token tensor converted by batch_converter, (batch_size, seq_len)
                sequence_len + 2 due to <cls> and <eos> token
                it could be (num_mutations, batch_size, seq_len) in case of multiple mutations
        """

        if x.dim() == 3:
            # evaluation on multiple mutations, x: (mutation_num, batch_size, seq_len)
            mutation_num, batch_size, seq_len = x.size()
            x = x.reshape(-1, seq_len)  # (mutation_num * batch_size, seq_len)
            reshape_needed = True
        else:
            # evaluation on single mutation, x: (batch_size, seq_len)
            reshape_needed = False
        
        output = self.ESM2_model(x, repr_layers=[self.ESM2_num_layers], return_contacts=False)
        # (batch_size, seq_len, embed_dim) or # (mutation_num*batch_size, seq_len, embed_dim)
        # slice 1:-1 to exclude <cls> and <eos> from pooling
        representations = output["representations"][self.ESM2_num_layers][:,1:-1,:] 
        # (batch_size, embed_dim) or # (mutation_num*batch_size, embed_dim)
        pooled_representations = representations.mean(1)
        logits = self.linear2(self.relu(self.linear1(pooled_representations)))

        if reshape_needed:
            return logits.view(mutation_num, batch_size, -1)  # (mutation_num, batch_size, 1)
        else:
            return logits # (batch_size, 1))
    

class ESM2AttentionHead(torch.nn.Module):
        
    def __init__(self, ESM2_model, output_dim):
        super(ESM2AttentionHead, self).__init__()
        
        self.ESM2_model = copy.deepcopy(ESM2_model)
        self.embed_dim = self.ESM2_model.embed_dim
        self.ESM2_num_layers = self.ESM2_model.num_layers
        self.attention1d = Attention1d(in_dim=self.embed_dim) 
        self.linear1 = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.embed_dim, output_dim)
    
    def forward(self, x):
        """
        TODO: this mean pooling does not work on sequences that have different lengths due to padding tokens
                mean pooling of sequences that have different lengths

        args
        ----
            x: token tensor converted by batch_converter, (batch_size, seq_len)
                sequence_len + 2 due to <cls> and <eos> token
                it could be (num_mutations, batch_size, seq_len) in case of multiple mutations
        """

        if x.dim() == 3:
            # evaluation on multiple mutations, x: (mutation_num, batch_size, seq_len)
            mutation_num, batch_size, seq_len = x.size()
            x = x.reshape(-1, seq_len)  # (mutation_num * batch_size, seq_len)
            reshape_needed = True
        else:
            # evaluation on single mutation, x: (batch_size, seq_len)
            reshape_needed = False
        
        output = self.ESM2_model(x, repr_layers=[self.ESM2_num_layers], return_contacts=False)
        # (batch_size, seq_len, embed_dim) or # (mutation_num*batch_size, seq_len, embed_dim)
        # slice 1:-1 to exclude <cls> and <eos> from pooling
        representations = output["representations"][self.ESM2_num_layers][:,1:-1,:]
        # (batch_size, embed_dim) or # (mutation_num*batch_size, embed_dim)
        pooled_representations = self.attention1d(representations, input_mask=None) 
        
        logits = self.linear2(self.relu(self.linear1(pooled_representations)))

        if reshape_needed:
            return logits.view(mutation_num, batch_size, -1)  # (mutation_num, batch_size, 1)
        else:
            return logits 
        

class ESM2MutationPositionHead(torch.nn.Module):
        
    def __init__(self, ESM2_model, output_dim):
        super(ESM2MutationPositionHead, self).__init__()
        
        self.ESM2_model = copy.deepcopy(ESM2_model)
        self.embed_dim = self.ESM2_model.embed_dim
        self.ESM2_num_layers = self.ESM2_model.num_layers
        self.linear1 = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.embed_dim, output_dim)
    
    def forward(self, x, mutation_position):
        """
        args
        ----
            x: token tensor converted by batch_converter, (batch_size, seq_len)
                sequence_len + 2 due to <cls> and <eos> token
                it could be (num_mutations, batch_size, seq_len) in case of multiple mutations
            mutation_position: position of the mutation, (batch_size, mutation_num)
        """

        output = self.ESM2_model(x, repr_layers=[self.ESM2_num_layers], return_contacts=False)
        # (batch_size, seq_len, embed_dim)
        representations = output["representations"][self.ESM2_num_layers] 
        batch_indices = torch.arange(x.size(0)).unsqueeze(1).expand_as(mutation_position)
        # representations[batch_indices, mutation_position, :], (batch_size, num_mutations, embed_dim)
        # pooled_representations, (batch_size, embed_dim)
        pooled_representations = representations[batch_indices, mutation_position, :].mean(dim=1) # (batch_size, embed_dim)
        logits = self.linear2(self.relu(self.linear1(pooled_representations)))

        return logits

