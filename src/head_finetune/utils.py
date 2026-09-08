import torch
import copy


def compute_general_mutation_score(model, ESM_model, mt_seq):
    """
    compute general mutation score for AttentionHead and MeanPoolingHead

    args
    ----
        model: AttentionHead or MeanPoolingHead
        ESM_model: pretrained ESM model
        mt_seq: encoded mutated sequences, (batch_size, seq_len)
    """

    output = ESM_model(mt_seq, repr_layers=[ESM_model.num_layers])
    x_representation = output["representations"][ESM_model.num_layers]

    return model(x_representation) # (batch_size, 1)


def compute_additive_mutation_score(model, ESM_model, mt_seq, mutation, batch_converter):
    '''
    compute additive mutation score for AttentionHead and MeanPoolingHead

    NOTE about mutation position
    batch_converter adds <cls> token at the first position and <eos> token at the last position
    position in the sequence data dict, wildtype amino acid + position + mutated amino acid, begins with 0.
    Therefore, we need to adjust the insertion of the <cls> token positioned at the first position.

    args
    ----
        model: AttentionHead or MeanPoolingHead
        ESM_model: pretrained ESM model
        mt_seq: encoded mutated sequences, (batch_size, seq_len)
        mutation: str of mutantation e.g., "A8K", (batch_size, num_mutations)
        batch_converter: batch converter objective
    
    returns
    -------
        additive_score: mutational proxy score
    '''
    device = mt_seq.device

    num_mutations = len(mutation[0])
    masked_seq = mt_seq.clone().unsqueeze(0).expand(num_mutations, -1, -1) # (number of mutations, batch_size, seq_len)
    mask_idx = batch_converter.alphabet.mask_idx
    mutation_position = get_mutation_position(mutation, adjust_position=1)

    batch_size, seq_len = mt_seq.size()

    for i in range(batch_size):
        
        masked_seq = masked_seq.clone()
        masked_seq[:, i, mutation_position[i]] = mask_idx
        
        for k in range(mutation_position.size(1)): # iterative over the number of mutations
            # masking all other mutated positions except mutation_position[i][k]
            masked_seq[k, i, mutation_position[i][k]] = int(mt_seq[i, mutation_position[i][k]])

    masked_seq = masked_seq.view(num_mutations*batch_size,seq_len).to(device)
    # (num_mutations*batch_size, seq_len, embed_dim)
    x_rep = ESM_model(masked_seq, repr_layers=[ESM_model.num_layers])["representations"][ESM_model.num_layers]
    x_rep = x_rep.reshape(num_mutations, batch_size, seq_len, -1) # (num_mutations, batch_size, seq_len, embed_dim)
    scores = model(x_rep) # (num_mutations, batch_size, 1)
    additive_scores = scores.sum(dim=0)

    return additive_scores # (batch_size, 1)


def get_mutation_position(mutation_batch, adjust_position):
    """
    args
    ----
        mutation_batch: 

    returns
    -------
        mutation_position: list of position of the mutation. this should only apply to single mutation, (batch_size, 1)
    """

    mutation_position_list = []

    for i in range(len(mutation_batch)):

        mutation_list_i = mutation_batch[i]
        mutation_position_list_i = []
        
        if isinstance(mutation_list_i, str):
            mutation_position_list_i.append(int(mutation_list_i[1:-1]))
            
        if isinstance(mutation_list_i, tuple):
            for mutation in mutation_list_i:
                mutation_position_list_i.append(int(mutation[1:-1]))

        mutation_position_list.append(mutation_position_list_i)

    return torch.tensor(mutation_position_list) + adjust_position


def compute_general_mutation_score_position_pooling(model, ESM_model, mt_seq, mutation):
    """
    compute general mutation score for MutationPositionHead

    args
    ----
        model: ESM2MutationPositionHead
        ESM_model: pretrained ESM model
        mt_seq: encoded mutated sequences, (batch_size, seq_len)
        mutation: str of mutantation e.g., "A8K", (batch_size, num_mutations)
    """
    device = mt_seq.device

    output = ESM_model(mt_seq, repr_layers=[ESM_model.num_layers])
    x_representation = output["representations"][ESM_model.num_layers] # (batch_size, seq_len, embed_dim)

    mutation_position = get_mutation_position(mutation, adjust_position=1) # (batch_size, num_mutations)
    mutation_position = mutation_position.to(device)

    return model(x_representation, mutation_position)  # (batch_size, 1)


def compute_additive_mutation_score_position_pooling(model, ESM_model, mt_seq, mutation, batch_converter):
    '''
    compute additive mutation score for MutationPositionHead

    args
    ----
        model: ESM2MutationPositionHead
        ESM_model: pretrained ESM model
        mt_seq: encoded mutated sequences, (batch_size, seq_len)
        mutation: str of mutantation e.g., "A8K", (batch_size, num_mutations)
        batch_converter: batch converter objective
    returns
    -------
        score: mutational proxy score
    '''
    device = mt_seq.device

    num_mutations = len(mutation[0])
    masked_seq = mt_seq.clone().unsqueeze(0).expand(num_mutations, -1, -1) # (number of mutations, batch_size, seq_len)
    mask_idx = batch_converter.alphabet.mask_idx
    mutation_position = get_mutation_position(mutation, adjust_position=1)

    batch_size, seq_len = mt_seq.size()

    for i in range(batch_size):
        
        masked_seq = masked_seq.clone()
        masked_seq[:, i, mutation_position[i]] = mask_idx
        
        for k in range(mutation_position.size(1)): # iterative over the number of mutations
            # masking all other mutated positions except mutation_position[i][k]
            masked_seq[k, i, mutation_position[i][k]] = int(mt_seq[i, mutation_position[i][k]])

    masked_seq = masked_seq.view(num_mutations*batch_size,seq_len).to(device)
    # (num_mutations*batch_size, seq_len, embed_dim)
    x_rep = ESM_model(masked_seq, repr_layers=[ESM_model.num_layers])["representations"][ESM_model.num_layers]
    x_rep = x_rep.reshape(num_mutations, batch_size, seq_len, -1) # (num_mutations, batch_size, seq_len, embed_dim)

    # TODO: there should be better way than looping
    additive_scores = torch.zeros((num_mutations, batch_size, 1))

    for k in range(num_mutations): # iterative over the number of mutations
        scores = model(x_rep[k, :, :, :], mutation_position[:,k].view(batch_size,1))
        additive_scores[k,:,:] = copy.copy(scores)

    additive_scores = additive_scores.sum(0)

    return additive_scores
