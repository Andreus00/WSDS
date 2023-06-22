import torch

coarse_path = 'data/coarse-grained'
fine_path = 'data/fine-grained'
mapping_path = 'data/map'

sentence_model_name = 'bert-base-cased'
gloss_model_name = 'bert-base-cased'
max_len = 512
batch_size = 4
epochs = 10
lr = 1e-5
weight_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gradient_clipping = 1.0

padding_token = -100


hyperparameter_grid = {
    'lr': [2e-5, 3e-5, 5e-5],
    'batch_size': [8, 16],
    'gradient_clipping': [0.5, 1.0, 1.5]
}

