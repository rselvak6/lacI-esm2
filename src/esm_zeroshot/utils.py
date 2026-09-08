import torch
import math
from torch.nn import Parameter
from torch.nn.init import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    uniform_,
    normal_,
    constant_,
    ones_,
    zeros_,
    _calculate_fan_in_and_fan_out,
)


def cal_bound(model: torch.nn.Module, layer_name: str):
    """Return bound for reinit given model and layer name"""
    assert "bias" in layer_name, f"no bias in {layer_name}"
    fan_in, _ = _calculate_fan_in_and_fan_out(
        model.state_dict()[layer_name.replace("bias", "weight")]
    )
    return 1 / math.sqrt(fan_in) if fan_in > 0 else 0


def init_esm_weight(model):
    
    for layer_name, p in model.state_dict().items():
        if "_proj" in layer_name:
            
            if "weight" in layer_name:
                
                if "out" in layer_name:
                    xavier_uniform_(p)
                
                else:
                    xavier_uniform_(p, gain=1 / math.sqrt(2))
                
            elif "bias" in layer_name:
                if "out" in layer_name:
                    constant_(p, 0.0)
                else:
                    bound = cal_bound(model=model, layer_name=layer_name)
                    uniform_(p, -bound, bound)
                        
        if "layer_norm" in layer_name:
            if "weight" in layer_name:
                Parameter(torch.ones_like(p))
            elif "bias" in layer_name:
                Parameter(torch.zeros_like(p))
                    
            
        if ("layers" and "fc" in layer_name) or ("contact_head" in layer_name):
            if "weight" in layer_name:
                kaiming_uniform_(p, a=math.sqrt(5))
            elif "bias" in layer_name:
                bound = cal_bound(model=model, layer_name=layer_name)
                uniform_(p, -bound, bound)
                    
        if "embed_positions" in layer_name:
            normal_(p)

        if layer_name == "lm_head.weight":
            xavier_uniform_(p)

        if layer_name == "lm_head.bias" or "lm_head.layer_norm.bias":
            Parameter(torch.zeros_like(p))

        if "dense" in layer_name:
            if "weight" in layer_name:
                kaiming_uniform_(p, a=math.sqrt(5))
            elif "bias" in layer_name:
                bound = cal_bound(model=model, layer_name=layer_name)
                uniform_(p, -bound, bound)
                
    return model


def compute_zeroshot_mutation_score(mutation, wt_sequence, token_probs, alphabet):
    # split mutation into wildtype_residue, position, mutation_residue
    """
    refer Language models enable zero-shot prediction of the effects of mutations on protein function Meier et al.,
    for score computation methods
    """


    if isinstance(mutation, str):
        # single mutation
        wt_aa, pos, mt_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
        pos -= 1

        if wt_sequence[pos] != wt_aa:
            raise ValueError("The listed wildtype does not match the provided sequence")

        wt_encoded, mt_encoded = alphabet.get_idx(wt_aa), alphabet.get_idx(mt_aa)

        # the first position in the sequence is for <CLS>
        # compute wildtype marginal probability
        # log-likelihood of mutation - log-likelihood of wildtype
        # refer to Language models enable zero-shot prediction of the effects of mutations on protein function Meier et al.,
        # for score computation methods   
        score = token_probs[0, pos+1, mt_encoded] - token_probs[0, pos+1, wt_encoded]
        score = score.item()

    elif isinstance(mutation, tuple):
        # multiple mutation

        score = 0

        for mt in mutation:
            wt_aa, pos, mt_aa = mt[0], int(mt[1:-1]), mt[-1]
            
            if wt_sequence[pos] != wt_aa:
                raise ValueError("The listed wildtype does not match the provided sequence")

            wt_encoded, mt_encoded = alphabet.get_idx(wt_aa), alphabet.get_idx(mt_aa)

            # the first position in the sequence is for <CLS>
            # compute wildtype marginal probability
            # log-likelihood of mutation - log-likelihood of wildtype
            # refer to Language models enable zero-shot prediction of the effects of mutations on protein function Meier et al.,
            # for score computation methods   
            this_score = token_probs[0, pos+1, mt_encoded] - token_probs[0, pos+1, wt_encoded]
            score += this_score.item()
        
    return score