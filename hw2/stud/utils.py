import torch
import hw2.stud.config as config
from hw2.stud.data_loader import DataTuple

def collate_fn(batch):
    
    if len(batch) < 8:
        dt = batch[0]
        batch.append(DataTuple(dt.encoded_sequence, 
                               dt.encoded_sequence_attention_mask, 
                               dt.senquence_output_mask,
                               dt.gold_sense_ids,
                               dt.gold_senses_idx,
                               dt.encoded_candidates,
                               dt.encoded_candidates_attention_mask))
    encoded_sequence = torch.stack([item.encoded_sequence for item in batch])
    encoded_sequence_attention_mask = torch.stack([item.encoded_sequence_attention_mask for item in batch])
    encoded_sequence_output_mask = torch.stack([item.senquence_output_mask for item in batch])
    gold_sense_ids = [item.gold_sense_ids for item in batch]
    gold_senses_idx = [item.gold_senses_idx for item in batch]
    max_len_candidates = max([item.encoded_candidates_attention_mask.shape[1] for item in batch])
    encoded_candidates = torch.zeros((len(batch), max_len_candidates, config.max_len), dtype=torch.long)
    encoded_candidates_attention_mask = torch.zeros_like(encoded_candidates)
    encoded_candidates_output_mask = torch.zeros((len(batch), max_len_candidates))
    for el in range(len(batch)):
        encoded_candidates[el, :batch[el].encoded_candidates.shape[1], :] = batch[el].encoded_candidates
        encoded_candidates_attention_mask[el, :batch[el].encoded_candidates.shape[1], :] = batch[el].encoded_candidates_attention_mask
        for i in range(batch[el].encoded_candidates.shape[1]):
            encoded_candidates_output_mask[el, i] = 1
    return encoded_sequence, encoded_sequence_attention_mask, encoded_sequence_output_mask, gold_sense_ids, gold_senses_idx, encoded_candidates, encoded_candidates_attention_mask, encoded_candidates_output_mask