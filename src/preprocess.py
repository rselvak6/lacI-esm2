import torch
import pandas as pd
import numpy as np
import esm
from tqdm import tqdm
from src.utils import *
from tape import TAPETokenizer, UniRepModel


def get_possible_mutations(wt_sequence, AA_candidates):
    """
        output: list of all possible mutations
                ['V0A', 'V0C', ...]
    """
    
    mutation_list = []

    for pos in range(len(wt_sequence)):
        
        wt_aa = wt_sequence[pos]
        
        for candidate_aa in AA_candidates:
            
            if wt_aa != candidate_aa:
                mut = wt_aa + str(pos) + candidate_aa
                mutation_list.append(mut)
            else:
                continue
    
    return mutation_list


def convert_mutation_to_full_sequence(mutation: str, wildtype_sequence: str):
    """
    convert mutation notation to the corresponding sequence based on the wildtype_sequence

    args
    ----
        mutation: mutation notation, for example "V0K". 
        NOTE: Index starts at 0.
        wildtype_sequence: wildtype protein sequence
    """
    
    wt_residue = mutation[0]
    position = int(mutation[1:-1])
    mt_residue = mutation[-1]
    
    return wildtype_sequence[:position] + mt_residue + wildtype_sequence[(position+1):], wt_residue, mt_residue, position


def build_representaton_data_dict(saving_dir: str, pretrained_model: str, AA2ID: dict, 
                                  target_sequence: str, batch_size: int, device):
    """
    build representation data dict using the selected ESM2 pretrained model
    TODO: saved data dict has bit too large file size

    Args
    ----
        target_sequence: sequence of the target protein, str
                        must be a full sequence without skipping of amino acids
                        for example, even though the first amino acid is not considered, it must be included 
                        otherwise, it leads to a wrong sequence ordering

    Returns
    -------
        representaton_data_dict: data dictionary with structure of 
                                {mutation sequence: output representation of the selected ESM2 model} 
    """
    
    representaton_data_dict = dict()
    
    print("loading pre-trained ESM2...")
    if pretrained_model == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    elif pretrained_model == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    elif pretrained_model == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    elif pretrained_model == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    elif pretrained_model == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        batch_converter = alphabet.get_batch_converter()
    else:
        raise ValueError("wrong pretrained model input")
    print("loading pre-trained ESM2 done")
    
    aa_candidate_list = list(AA2ID.values())
    possible_mutations_list = get_possible_mutations(target_sequence, aa_candidate_list)
    
    batch = []
    model.to(device)
    model.eval()
    for mutation in tqdm(possible_mutations_list):
        
        mut_seq, wt_residue, mt_residue, position = convert_mutation_to_full_sequence(mutation, target_sequence)
        batch.append((mutation, mut_seq))
        
        if len(batch) == batch_size:
            with torch.no_grad():
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                batch_tokens = batch_tokens.to(device)
                outputs = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
                batch_representations = outputs["representations"][model.num_layers]
            
            for i in range(batch_size):
                batch_representations[i]
                representaton_data_dict[batch[i][1]] = batch_representations[i].detach().cpu().numpy()
            
            batch = []

    # process the last batch if it's smaller than batch_size
    if len(batch) > 0:
        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)
            outputs = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
            batch_representations = outputs["representations"][model.num_layers]
        
        for i in range(len(batch)):
            representaton_data_dict[batch[i][1]] = batch_representations[i].detach().cpu().numpy()
    
    save_data(saving_dir + pretrained_model + "_single_mutation_representaton_data_dict.pkl", representaton_data_dict)


def WT_heatmap_to_mutation_label_dict(heatmap: pd.DataFrame, sequence: str, mode: str):
    """
    convert heatmap based on the experimental data to a dictionary

    args
    ----
        heatmap: heatmap read by using pd.read_csv()
        sequence: target protein sequence
        mode: assign bin number based on mode

    returns
    -------
        mutation_label_dict: {mutation : label}
    """
    
    mutation_label_dict = dict()
    mutation_candidate = list(heatmap.index)
    
    a_count = 0
    r_count = 0
    m_count = 0
    s_count = 0
    misc_count = 0
    
    for i, aa in enumerate(sequence):
        
        try:
            mutation_results = heatmap["{}".format(i+1)] # enumeration index starts at 0, heatmap index starts at 1
        except:
            continue # mutation result starts at the second position, since we ignore the first and the last amino acids 
        
        for mc in mutation_candidate:
            
            mutation_result = mutation_results[mc]

            if mode == 'rep':
                
                if mutation_result.startswith("M"):
                    # non-functional
                    label_mutation_result = -1
                    m_count += 1
                    
                elif mutation_result.startswith("S"):
                    # ignoring S for now
                    label_mutation_result = 0
                    s_count += 1
                    
                elif mutation_result.startswith("R"):
                    # repressor phenotype
                    # functional
                    label_mutation_result = int(mutation_result[1:])
                    r_count += 1
                    
                elif mutation_result.startswith("A"):
                    # ignoring A for now
                    label_mutation_result = 0
                    a_count += 1
                else:
                    # missing data or wildtype
                    label_mutation_result = 0 
                    misc_count += 1
                    
            elif mode == 'antirep':
                
                if mutation_result.startswith("M"):
                    # non-functional
                    label_mutation_result = -1
                    m_count += 1
                    
                elif mutation_result.startswith("S"):
                    # ignoring S for now
                    label_mutation_result = 0
                    s_count += 1
                    
                elif mutation_result.startswith("A"):
                    # antirepressor phenotype
                    # functional
                    label_mutation_result = int(mutation_result[1:])
                    a_count += 1
                    
                elif mutation_result.startswith("R"):
                    # ignoring R for now
                    label_mutation_result = 0
                    r_count += 1
                else:
                    # missing data or wildtype
                    label_mutation_result = 0 
                    misc_count += 1
            
            mutation = aa + str(i) + mc
            mutation_label_dict[mutation] = label_mutation_result
            
    print("A:{}".format(a_count))
    print("R:{}".format(r_count))
    print("S:{}".format(s_count))
    print("M:{}".format(m_count))
    print("Misc:{}".format(misc_count))
    
    return mutation_label_dict


def build_sequence_data_dict(heatmap_dir, wildtype_sequence, task, mode, remove_stop_codon=True):

    """
    build sequence data dict using heatmap

    returns
    -------
        sequence_data_dict: {mutated sequence : (mutation, binary label, bin label)}
    """
    heatmap = pd.read_csv(heatmap_dir, index_col=0)
    converted_data = dict()
    stop_codon_count = 0
    missing_count = 0

    mutation_label_dict = WT_heatmap_to_mutation_label_dict(heatmap, wildtype_sequence, mode)
    
    for mutation, label in mutation_label_dict.items():
        full_mutation_sequence, wt_residue, mt_residue, position = convert_mutation_to_full_sequence(mutation, wildtype_sequence)
        
        if remove_stop_codon:
            if mt_residue == "*":
                stop_codon_count += 1
                continue
            elif wt_residue == "*":
                stop_codon_count += 1
                continue
            elif full_mutation_sequence == wildtype_sequence:
                continue
            else:
                converted_data[full_mutation_sequence] = (mutation, label) # exclude the end and first residue

    filtered_data = dict()

    for mutated_sequence, mt_label_pair in converted_data.items():
        
        mutation, label = mt_label_pair
        
        if label != 0:
            
            if task == "binary":
                # non-functional to 0, functional (A or R) to 1
                label = int(np.clip(label, 0, 1)) # binary label
                filtered_data[mutated_sequence] = (mutation, label)
            
            elif task == "score":
                if label < 0:
                    #repressors/antirepressors: 1-10
                    filtered_data[mutated_sequence] = (mutation, label)
                if label > 0:
                    #M becomes 0
                    filtered_data[mutated_sequence] = (mutation, label + 1)
                    
        else:
            # remove data with label 0 (missing data and wildtype)
            missing_count += 1
            continue
            
    print("stop codon count : {}".format(stop_codon_count))
    print("missing count : {}".format(missing_count))
    print("data count : {}".format(len(filtered_data)))
    
    return filtered_data


def WT_doubles_heatmap_to_sequence_data_dict(heatmap_dir, frac_threshold, task):

    if task not in ["binary", "score"]:
        raise ValueError("wrong task assigned")
    
    heatmap = pd.read_csv(heatmap_dir)
    
    a_count = 0
    r_count = 0
    m_count = 0
    s_count = 0
    misc_count = 0
    position_exclusion_count = 0
    frac_exclusion_count = 0

    sequence_data = dict()
    
    for i in range(heatmap.shape[0]):
        
        row_i = heatmap.loc[i]
        sequence = row_i["sequence"][:-1] # excluding *
        first_mutation = row_i["First"]
        first_mutation_aa = row_i["First"][-1]
        second_mutation = row_i["Second"]
        second_mutation_aa = row_i["Second"][-1]
        first_mutation_position = int(first_mutation[1:-1]) - 1 # index starts at 1, therefore -1 to make it to start at 0
        second_mutation_position = int(second_mutation[1:-1]) - 1 # index starts at 1, therefore -1 to make it to start at 0
        frac_max = row_i["frac_max"]
        mutation_result = row_i["max bin"]
        
        if first_mutation_position in [0,360]:
            position_exclusion_count += 1
            continue # exclude mutation on the first and the last position

        if second_mutation_position in [0,360]:
            position_exclusion_count += 1
            continue # exclude mutation on the first and the last position

        if first_mutation_aa == "*":
            position_exclusion_count += 1
            continue

        if second_mutation_aa == "*":
            position_exclusion_count += 1
            continue
        
        if frac_threshold:
            if frac_max < frac_threshold:
                frac_exclusion_count += 1
                continue
            
        if mutation_result.startswith("M"):
            # non-functional
            label_mutation_result = -1
            m_count += 1
                
        elif mutation_result.startswith("S"):
            # non-functional
            label_mutation_result = -1
            s_count += 1

        elif mutation_result.startswith("R"):
            # repressor phenotype
            # functional
            label_mutation_result = int(mutation_result[1:])
            r_count += 1

        elif mutation_result.startswith("A"):
            # anti-repressor phenotype
            # functional
            label_mutation_result = int(mutation_result[1:]) + 10
            a_count += 1
        else:
            # missing data or wildtype
            label_mutation_result = 0 
            misc_count += 1
        
        mutation = (first_mutation[0] + str(first_mutation_position) + first_mutation[-1],
                    second_mutation[0] + str(second_mutation_position) + second_mutation[-1])
        sequence_data[sequence] = (mutation, label_mutation_result)

    filtered_data = dict()
    anti_repressor_count_filtered = 0
    misc_count_filtered = 0

    for mutated_sequence, mt_label_pair in sequence_data.items():
        
        mutation, label = mt_label_pair
        
        if label != 0:
            
            if task == "binary":
                # non-functional to 0, functional (A,R) to 1
                label = int(np.clip(label, 0, 1)) # binary label: functional vs. non-functional
                filtered_data[mutated_sequence] = (mutation, label)
            
            elif task == "score":
                # non-functional to 0, else use repressor phenotype
                if label <= 10:
                    # only consider repressor phenotypes
                    # if label > 10, they are anti-repressor phenotypes
                    filtered_data[mutated_sequence] = (mutation, label)
                if label > 10:
                    anti_repressor_count_filtered += 1
                    
        else:
            # remove data with label 0 (missing data and wildtype)
            misc_count_filtered += 1
        
    print("total data points: {}".format(heatmap.shape[0]))
    print("available data points: {}".format(len(filtered_data)))
    print("excluded data points due to positions: {}".format(position_exclusion_count))
    print("excluded data points due to frac threshold: {}".format(frac_exclusion_count))
    
    print("Within the total data points")
    print("A:{}".format(a_count))
    print("R:{}".format(r_count))
    print("S:{}".format(s_count))
    print("M:{}".format(m_count))
    print("Misc:{}".format(misc_count))

    print("Within the available data points")
    print("A:{}".format(anti_repressor_count_filtered))
    print("Misc:{}".format(misc_count_filtered))
    
    return filtered_data


def batch_convert_unirep(tokenizer: TAPETokenizer, mutation_sequence_pair_batch):
    """
    encode SequenceDataset batched output using TAPETokenizer
    this is required since TAPETokenizer is not compatible to encode the batched output of SequenceDataset
    """

    encoded_sequence_list = []

    for mut, seq in mutation_sequence_pair_batch:
        encoded_sequence_list.append(tokenizer.encode(seq))

    return torch.from_numpy(np.array(encoded_sequence_list)).long()


def build_unirep_data_dict(saving_dir: str, pretrained_model: str, AA2ID: dict,
                           target_sequence: str, batch_size: int, device):
    """
    build unirep data dict using the pretrained Unirep model
    TODO: saved data dict has bit too large file size

    Args
    ----
        target_sequence: sequence of the target protein, str
                        must be a full sequence without skipping of amino acids
                        for example, even though the first amino acid is not considered, it must be included 
                        otherwise, it leads to a wrong sequence ordering

    Returns
    -------
        unirep_data_dict: data dictionary with structure of 
                                {mutation sequence: output representation of the selected Unirep model} 
    """
    
    unirep_data_dict = dict()
    
    print("loading pre-trained Unirep...")
    if pretrained_model == "unirep-1900":
        model = UniRepModel.from_pretrained("babbler-1900")
        tokenizer = TAPETokenizer(vocab='unirep')
    else:
        raise ValueError("wrong pretrained model input")
    print("loading pre-trained Unirep done")
    
    aa_candidate_list = list(AA2ID.keys())
    possible_mutations_list = get_possible_mutations(target_sequence, aa_candidate_list)
    
    batch = []
    model.to(device)
    model.eval()
    for mutation in tqdm(possible_mutations_list): 
        
        mut_seq, wt_residue, mt_residue, position = convert_mutation_to_full_sequence(mutation, target_sequence)
        batch.append((mutation, mut_seq))
        
        if len(batch) == batch_size:
            with torch.no_grad():
                batch_tokens = batch_convert_unirep(tokenizer, batch)
                batch_tokens = batch_tokens.to(device)
                # outputs[0] is hidden representations for all amino acids (batch_size, seq_len+2, embed_dim)
                # outputs[1] is concatenated representations of the final hidden state and cell state for each sequence,
                # (batch_size, embed_dim*2)
                outputs = model(batch_tokens)
                # averaged representations over all amino acids in the sequence
                batch_representations = outputs[0].mean(dim=1) # (batch_size, embed_dim)
            
            for i in range(batch_size):
                unirep_data_dict[batch[i][1]] = batch_representations[i].detach().cpu().numpy()
            
            batch = []

    # process the last batch if it's smaller than batch_size
    if len(batch) > 0:
        with torch.no_grad():
            batch_tokens = batch_convert_unirep(tokenizer, batch)
            batch_tokens = batch_tokens.to(device)
            outputs = model(batch_tokens)
            batch_representations = outputs[0].mean(dim=1) # (batch_size, embed_dim)
        
        for i in range(len(batch)):
            unirep_data_dict[batch[i][1]] = batch_representations[i].detach().cpu().numpy()
    
    save_data(saving_dir + pretrained_model + "_single_mutation_representaton_data_dict.pkl", unirep_data_dict)


def find_mutations(seq_wt, mutated_seq):
    """
    Identifies mutations between two protein sequences of the same length.

    Args:
        seq_a (str): Reference protein sequence.
        seq_b (str): Mutated protein sequence.

    Returns:
        list: List of mutations in the format "V0K".
    """
    return [f"{seq_wt[i]}{i}{mutated_seq[i]}" for i in range(len(seq_wt)) if seq_wt[i] != mutated_seq[i]]


def build_garuss_sequence_data_dict(data_dir, wildtype_sequence):
    """
    for the input data, refer to 
    https://github.com/churchlab/lac_repression/blob/main/machine_learning/a005_updated_tournament_of_champions_singles-regular_cv-clean.ipynb
    """
    
    single_mutation_df = pd.read_csv(data_dir, usecols=[0,1], header=None)
    sequence_data_dict = dict()
    
    for index, row in single_mutation_df.iterrows():
        mutated_seq = row[0]
        fitness_val = row[1]
        
        mutations = find_mutations(wildtype_sequence, mutated_seq)
        
        sequence_data_dict[mutated_seq] = (mutations[0], fitness_val)
        
    return sequence_data_dict