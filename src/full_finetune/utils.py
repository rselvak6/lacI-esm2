import torch


def compute_additive_mutation_score(model, mt_seq, mutation, batch_converter):
    '''
    compute additive mutation score

    NOTE about mutation position
    batch_converter adds <cls> token at the first position and <eos> token at the last position
    position in the sequence data dict, wildtype amino acid + position + mutated amino acid, begins with 0.
    Therefore, we need to adjust the insertion of the <cls> token positioned at the first position.

    args
    ----
        model: ESM model with head
        mt_seq: encoded mutated sequences, (batch_size, seq_len)
        mutation: str of mutantation e.g., "A8K", (batch_size, num_mutations)
        batch_converter: batch converter objective
    
    returns
    -------
        additive_score: mutational proxy score
    '''
    device = mt_seq.device

    masked_seq = mt_seq.clone().unsqueeze(0).expand(len(mutation[0]), -1, -1) # (number of mutations, batch_size, seq_len)
    mask_idx = batch_converter.alphabet.mask_idx
    mutation_position = get_mutation_position(mutation, adjust_position=1)

    batch_size = int(mt_seq.size(0))

    for i in range(batch_size):
        
        masked_seq = masked_seq.clone()
        masked_seq[:, i, mutation_position[i]] = mask_idx
        
        for k in range(mutation_position.size(1)): # iterative over the number of mutations
            # masking all other mutated positions except mutation_position[i][k]
            masked_seq[k, i, mutation_position[i][k]] = int(mt_seq[i, mutation_position[i][k]])

    masked_seq = masked_seq.to(device)
    scores = model(masked_seq)
    additive_scores = scores.sum(dim=0)

    return additive_scores


def get_mutation_position(mutation_batch, adjust_position):
    """
    args
    ----
        mutation_batch: 
        adjust_position: 

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


def compute_additive_mutation_score_position_pooling(model, mt_seq, mutation, batch_converter):
    '''
    NOTE about mutation position
    batch_converter adds <cls> token at the first position and <eos> token at the last position
    position in the sequence data dict, wildtype amino acid + position + mutated amino acid, begins with 0.
    Therefore, we need to adjust the insertion of the <cls> token positioned at the first position.

    args
    ----
        model: ESM2MutationPositionHead
        mt_seq: mutant seq
        mutation: str of mutantation e.g., "A8K"
    
    returns
    -------
        score: mutational proxy score
    '''
    device = mt_seq.device

    masked_seq = mt_seq.clone().unsqueeze(0).expand(len(mutation[0]), -1, -1) # (number of mutations, batch_size, seq_len)
    mask_idx = batch_converter.alphabet.mask_idx
    mutation_position = get_mutation_position(mutation, adjust_position=1)

    batch_size = int(mt_seq.size(0))

    for i in range(batch_size):
        masked_seq = masked_seq.clone()
        masked_seq[:, i, mutation_position[i]] = mask_idx
        
        for k in range(mutation_position.size(1)): # iterative over the number of mutations
            # masking all other mutated positions except mutation_position[i][k]
            masked_seq[k, i, mutation_position[i][k]] = int(mt_seq[i, mutation_position[i][k]])

    masked_seq = masked_seq.to(device)
    additive_scores = torch.zeros((mutation_position.size(1), batch_size, 1))

    # TODO: there should be better way than looping
    for k in range(mutation_position.size(1)): # iterative over the number of mutations
        scores = model(masked_seq[k, :, :], mutation_position[:,k].view(batch_size,1))
        additive_scores[k,:,:] = scores

    additive_scores = additive_scores.sum(0)

    return additive_scores

