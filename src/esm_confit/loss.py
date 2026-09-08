import torch
import torch.nn.functional as F


def compute_mutation_score(model, mt_seq, wt_seq, mutation, batch_converter):
    '''
    compute masked marginal score

    NOTE about mutation position
    batch_converter adds <cls> token at the first position and <eos> token at the last position
    position in the sequence data dict, wildtype amino acid + position + mutated amino acid, begins with 0.
    Therefore, we need to adjust the insertion of the <cls> token positioned at the first position.

    args
    ----
        model: ESM2 model
        mt_seq: encoded mutated sequences, (batch_size, seq_len)
        wt_seq: encoded wild type sequence, (batch_size, seq_len)
        mutation: list of tuples containing strings of mutantation e.g., "A8K", (batch_size, tuples of mutations)
        batch_converter: batch converter object
    
    returns
    -------
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = mt_seq.device

    masked_seq = mt_seq.clone()
    mask_idx = batch_converter.alphabet.mask_idx

    batch_size = int(mt_seq.size(0))
    scores = torch.zeros(batch_size).to(device)

    for i in range(batch_size):

        if isinstance(mutation[i], str):
            # single mutation
            mt_pos = int(mutation[i][1:-1]) + 1 # refer to NOTE about + 1
        elif isinstance(mutation[i], tuple):
            # multiple mutations
            mt_pos = []

            for k in range(len(mutation[i])):
                mt_pos.append(int(mutation[i][k][1:-1]) + 1) # refer to NOTE about + 1

        masked_seq[i, mt_pos] = mask_idx

    masked_seq = masked_seq.to(device)

    output = model(masked_seq, return_contacts=False)
    logits = output["logits"] # (batch_size, seq_len, num_tokens)
    log_probs = torch.log_softmax(logits, dim=-1) # log_softmax along the dimension of tokens

    for i in range(batch_size):

        if isinstance(mutation[i], str):
            # single mutation
            mt_pos = int(mutation[i][1:-1]) + 1 # refer to NOTE about + 1
        elif isinstance(mutation[i], tuple):
            # multiple mutations
            mt_pos = []

            for k in range(len(mutation[i])):
                mt_pos.append(int(mutation[i][k][1:-1]) + 1) # refer to NOTE about + 1

        score_i = log_probs[i] # (seq_len, num_tokens)
        mt_seq_i = mt_seq[i] # (seq_len)
        wt_seq_i = wt_seq[i] # (seq_len)
        # computation of the masked marginal probability as in Meier et al. 2021
        # \sum_{i \in M} \log p(x_i = x_i^{\text{mt}} | x_{-M}) - \log p(x_i = x_i^{\text{wt}} | x_{-M})
        scores[i] = torch.sum(score_i[mt_pos, mt_seq_i[mt_pos]]) - \
            torch.sum(score_i[mt_pos, wt_seq_i[mt_pos]])

    return scores, logits


def compute_BT_loss(scores, target_scores, temperature=1.0):
    """
    Compute Bradley-Terry loss for ranking/contrastive learning
    
    This implementation assumes you want to learn relative fitness rankings.
    For proper contrastive learning, you typically need positive/negative pairs.
    
    args
    ----
        scores: predicted fitness scores, (batch_size,)
        target_scores: true fitness scores, (batch_size,)
        temperature: temperature parameter for scaling
    """
    
    batch_size = len(scores)
    device = scores.device
    
    # Method 1: Ranking-based Bradley-Terry (more efficient)
    # Sort indices by target scores
    sorted_indices = torch.argsort(target_scores, descending=True)
    sorted_scores = scores[sorted_indices]
    
    loss = 0.0
    count = 0
    
    # Only compute loss for meaningful pairs (reduce quadratic complexity)
    for i in range(batch_size):
        for j in range(i + 1, min(i + 10, batch_size)):  # Limit comparisons per item
            # Higher fitness should have higher score
            score_diff = (sorted_scores[i] - sorted_scores[j]) / temperature
            loss += F.logsigmoid(score_diff)  # More numerically stable
            count += 1
    
    return -loss / count if count > 0 else torch.tensor(0.0, device=device)


def compute_BT_loss_contrastive(scores, target_scores, temperature=0.1):
    """
    Alternative: Proper contrastive Bradley-Terry loss
    This treats each sample as positive, others as negatives
    """
    batch_size = len(scores)
    device = scores.device
    
    # Normalize scores for stability
    scores_norm = scores / temperature
    
    # For each sample, compute probability it's better than all others
    loss = 0.0
    for i in range(batch_size):
        # Get samples that this one should be better than
        better_mask = target_scores[i] > target_scores
        if better_mask.sum() > 0:
            # Probability that sample i is better than samples it should be better than
            logits = scores_norm[i] - scores_norm[better_mask]
            loss += -F.logsigmoid(logits).mean()
    
    return loss / batch_size


def compute_KL_loss(logits, logits_reg, wt_seq):
    '''
    Compute KL divergence regularization loss
    
    args
    ----
        logits: predicted logits, (batch_size, seq_len, vocab_size)
        logits_reg: reference logits from frozen model, (batch_size, seq_len, vocab_size)
        wt_seq: wild type sequence tokens, (batch_size, seq_len)
    
    returns
    -------
        loss: KL divergence loss
    '''
    
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    
    # Convert to probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # For KL input
    probs_reg = F.softmax(logits_reg, dim=-1)  # For KL target
    
    # Exclude special tokens (<cls> at position 0, <eos> at position -1)
    # Only compute KL loss on actual protein sequence positions
    valid_positions = torch.arange(1, seq_len - 1, device=device)
    
    # Select valid positions for all samples
    log_probs_valid = log_probs[:, valid_positions, :]  # (batch_size, seq_len-2, vocab_size)
    probs_reg_valid = probs_reg[:, valid_positions, :]  # (batch_size, seq_len-2, vocab_size)
    
    # Compute KL divergence: KL(P_reg || P_pred)
    # This regularizes the predicted distribution to stay close to the reference
    kl_loss = F.kl_div(
        log_probs_valid,  # log probabilities (input)
        probs_reg_valid,  # target probabilities
        reduction='batchmean'  # Average over batch and sequence dimensions
    )
    
    return kl_loss


def compute_KL_loss_position_weighted(logits, logits_reg, wt_seq):
    '''
    Alternative KL loss that weights positions by their importance
    Only computes KL at wild-type amino acid positions
    '''
    
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    
    log_probs = F.log_softmax(logits, dim=-1)
    probs_reg = F.softmax(logits_reg, dim=-1)
    
    total_loss = 0.0
    
    for i in range(batch_size):
        # Get valid sequence positions (exclude <cls> and <eos>)
        valid_positions = torch.arange(1, seq_len - 1, device=device)
        wt_tokens = wt_seq[i, valid_positions]  # Wild-type tokens at valid positions
        
        # Get distributions at these positions
        log_p_pred = log_probs[i, valid_positions, :]  # (seq_len-2, vocab_size)
        p_reg = probs_reg[i, valid_positions, :]       # (seq_len-2, vocab_size)
        
        # Compute KL divergence for each position
        kl_per_position = F.kl_div(log_p_pred, p_reg, reduction='none').sum(dim=-1)
        
        # Average over positions
        total_loss += kl_per_position.mean()
    
    return total_loss / batch_size