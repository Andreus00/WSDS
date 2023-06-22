import torch
from torch.utils.data import DataLoader, IterableDataset
from torch import LongTensor
from transformers import AutoTokenizer
from typing import List, Dict, NamedTuple, Optional, Tuple
from transformers.tokenization_utils_base import BatchEncoding
import os
import json
import hw2.stud.config as config



class DataTuple(NamedTuple):
    encoded_sequence: torch.LongTensor  # [CLS] + text + [SEP]
    encoded_sequence_attention_mask: torch.LongTensor # 1 for text, 0 for padding
    senquence_output_mask: List[torch.Tensor]  # List of tensors, each tensor is a mask that indicates which token is a target word
    gold_sense_ids: List[str]  # gold sense ids
    gold_senses_idx: List[int]  # index of the gold sense in the candidates list
    encoded_candidates: List[torch.LongTensor]  # [CLS] + candidate + [SEP]
    encoded_candidates_attention_mask: List[torch.LongTensor] # 1 for candidate, 0 for padding



class WSDDatasetLoader:

    def __init__(self, coarse_path, fine_path, mapping_path, sentence_tokenizer, gloss_tokenizer, max_len):
        self.coarse_path = coarse_path
        self.fine_path = fine_path
        self.mapping_path = mapping_path
        self.sentence_tokenizer = sentence_tokenizer
        self.gloss_tokenizer = gloss_tokenizer
        self.max_len = max_len
        self.all_senses = {}
        self.gloss_id_to_gloss = {}
        self.load_data()
        self.init_gloss_id_to_gloss()
        self.process_data()

    def load_data(self):
        # load data from the data_path
        names = ['coarse', 'fine', 'mapping']
        for idx, path in enumerate([self.coarse_path, self.fine_path, self.mapping_path]):
            files: List[str] = os.listdir(path)
            for file in files:
                # open the file as a json
                with open(os.path.join(path, file), 'r') as f:
                    json_data = json.load(f)
                    filename_prefix = file.split('_')[0]
                    setattr(self, f"{names[idx]}_{filename_prefix}", json_data)
    
    def init_gloss_id_to_gloss(self):
        for hypernym_id, glosses_list in self.mapping_coarse.items():
            for item in glosses_list:
                for gloss_id, gloss in item.items():
                    self.gloss_id_to_gloss[gloss_id] = gloss


    def map_coarse_to_fine(self, candidates_ids):
        return [self.gloss_id_to_gloss[cand_id] for cand_id in candidates_ids]


    def pack_data(self, values):
        '''
        Pack the data into a list of DataTuple
        '''
        sequence = " ".join(values["words"])
        batch_encoding = self.sentence_tokenizer(sequence, add_special_tokens=True, padding="max_length", max_length=self.max_len, truncation=True, return_length=True, return_tensors='pt')
        encoded_sequence = batch_encoding["input_ids"]
        encoded_sequence_attention_mask = batch_encoding["attention_mask"]
        senses = values["senses"]
        senses_target_idx, gold_sense_ids = zip(*senses.items())
        senses_target_idx = [int(x) for x in senses_target_idx]

        senses_new_ids = batch_encoding.word_ids()
        encoding_length = batch_encoding["length"]
        candidates = values["candidates"]

        data_tuples = []


        for target_idx, canddidate_ids in candidates.items():
            _target_idx = int(target_idx)
            senquence_output_mask = torch.zeros_like(encoded_sequence)
            for j in range(_target_idx, encoding_length):
                if senses_new_ids[j] == _target_idx:
                    senquence_output_mask[0, j] = 1
            gold = senses[target_idx]
            gold_idx = canddidate_ids.index(gold[0])
            gold_tensor = torch.zeros((1, len(canddidate_ids)))
            gold_tensor[0, gold_idx] = 1
            candidate_glosses = self.map_coarse_to_fine(canddidate_ids)
            cand_batch_encoding = self.gloss_tokenizer(candidate_glosses, add_special_tokens=True, padding='max_length', max_length=self.max_len, truncation=True, return_length=True, return_tensors='pt')
            data_tuples.append(DataTuple(encoded_sequence, encoded_sequence_attention_mask, senquence_output_mask, gold, gold_tensor, cand_batch_encoding["input_ids"].unsqueeze(0), cand_batch_encoding["attention_mask"].unsqueeze(0)))

        return data_tuples

    def process_data(self):
        # iterate on the fine-grained data
        # for each one, I need to create a DataTuple
        # I need to tokenize the text and the candidates
        # I need to find the index of the gold sense in the candidates
        names = ['train', 'val', 'test']
        for name in names:
            setattr(self, f"{name}_data", [])
            data = getattr(self, f"{name}_data")
            for id, values in getattr(self, f"fine_{name}").items():
                data.extend(self.pack_data(values))

    def get_datasets(self):
        return WSDDataset(self.train_data), WSDDataset(self.val_data), WSDDataset(self.test_data)


    def tokenize_multiple_sequences(self, sequences):
        return self.tokenizer(sequences, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def tokenize_from_list_of_tokens(self, tokens):
        text = " ".join(tokens)
        return self.tokenizer(text)


class WSDDataset(IterableDataset):

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)



def main():
    coarse_path = 'data/coarse-grained'
    fine_path = 'data/fine-grained'
    mapping_path = 'data/map'
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = WSDDatasetLoader(coarse_path, fine_path, mapping_path, tokenizer, config.max_len)
    train, val, test = dataset.get_datasets()
    for el in train:
        print(el)
        break
