import torch
from src.utils import *
import torch.nn.functional as F


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
    
    
class MeanPoolingHead(torch.nn.Module):
    """
    MLP head on mean pooled sequence of input representations
    """
    
    def __init__(self, input_dim, output_dim):
        super(MeanPoolingHead, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.input_dim, output_dim)
    
    def forward(self, x):
        """
        TODO: this mean pooling does not work on sequences that have different lengths due to padding tokens
                
        args
        ----
            x: protein representations output by ESM, (batch_size, seq_len, embed_dim)
                sequence_len + 2 due to <cls> and <eos> token
                it could be (num_mutations, batch_size, seq_len, embed_dim) in case of multiple mutations
        """

        if x.dim() == 4:
            # evaluation on multiple mutations, x: (mutation_num, batch_size, seq_len, embed_dim)
            mutation_num, batch_size, seq_len, embed_dim = x.size()
            x = x.reshape(mutation_num*batch_size, seq_len, embed_dim)  # (mutation_num * batch_size, seq_len, embed_dim)
            reshape_needed = True
        else:
            # evaluation on single mutation, x: (batch_size, seq_len, embed_dim)
            reshape_needed = False
        
        # slice 1:-1 to exclude <cls> and <eos> from pooling
        # (batch_size, embed_dim) or # (mutation_num*batch_size, embed_dim)
        pooled_representation = x[:,1:-1,:].mean(1)
        logits = self.linear2(self.relu(self.linear1(pooled_representation)))
    
        if reshape_needed:
            return logits.view(mutation_num, batch_size, -1)  # (mutation_num, batch_size, 1)
        else:
            return logits # (batch_size, 1)
    
    
class AttentionHead(torch.nn.Module):
    """
    MLP head on attention pooled representation
    """
        
    def __init__(self, input_dim, output_dim):
        super(AttentionHead, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention1d = Attention1d(in_dim=self.input_dim) 
        self.linear1 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.input_dim, output_dim)

    
    def forward(self, x):
        """
        TODO: this attention pooling does not work on sequences that have different lengths due to padding tokens
                
        args
        ----
            x: protein representations output by ESM, (batch_size, seq_len, embed_dim)
                sequence_len + 2 due to <cls> and <eos> token
                it could be (num_mutations, batch_size, seq_len, embed_dim) in case of multiple mutations
        """

        if x.dim() == 4:
            # evaluation on multiple mutations, x: (mutation_num, batch_size, seq_len, embed_dim)
            mutation_num, batch_size, seq_len, embed_dim = x.size()
            x = x.reshape(mutation_num*batch_size, seq_len, embed_dim)  # (mutation_num * batch_size, seq_len, embed_dim)
            reshape_needed = True
        else:
            # evaluation on single mutation, x: (batch_size, seq_len, embed_dim)
            reshape_needed = False
        
        # slice 1:-1 to exclude <cls> and <eos> from pooling
        # (batch_size, embed_dim) or # (mutation_num*batch_size, embed_dim)
        pooled_representation = self.attention1d(x[:,1:-1,:], input_mask=None)
        logits = self.linear2(self.relu(self.linear1(pooled_representation)))

        if reshape_needed:
            return logits.view(mutation_num, batch_size, -1)  # (mutation_num, batch_size, 1)
        else:
            return logits # (batch_size, 1)
    
    
class MutationPositionHead(torch.nn.Module):
        
    def __init__(self, input_dim, output_dim):
        super(MutationPositionHead, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.input_dim, output_dim)

    def forward(self, x, mutation_position):
        """
        args
        ----
            x: protein representations output by ESM, (batch_size, seq_len, embed_dim)
                sequence_len + 2 due to <cls> and <eos> token
            mutation_position: position of the mutation, (batch_size, num_mutations)
        """
        batch_indices = torch.arange(x.size(0)).unsqueeze(1) # (batch_size, 1)
        pooled_representations = x[batch_indices, mutation_position, :].mean(dim=1) # (batch_size, embed_dim)
        logits = self.linear2(self.relu(self.linear1(pooled_representations))) # (batch_size, 1)

        return logits