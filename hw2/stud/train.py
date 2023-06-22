from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch import LongTensor
import hw2.stud.config as config
from hw2.stud.biencoder import BiEncoder
from hw2.stud.data_loader import WSDDatasetLoader
from hw2.stud.utils import collate_fn
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def sweep_model():
    # logger
    wandb_logger = WandbLogger(project="nlp2023-hw2")
    wandb_config = wandb.config

    sentence_model_tokenizer = AutoTokenizer.from_pretrained(wandb_config["sentence_model_name"])
    gloss_model_tokenizer = AutoTokenizer.from_pretrained(wandb_config["gloss_model_name"])
    # data
    data = WSDDatasetLoader(config.coarse_path, config.fine_path, config.mapping_path, sentence_model_tokenizer, gloss_model_tokenizer, wandb_config["max_len"])
    train_dataset, val_dataset, test_dataset = data.get_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    # model
    module = BiEncoder(wandb_config)

    # wandb watch
    wandb_logger.watch(module)


    # checkpointer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='model',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    trainer = pl.Trainer(accelerator="cuda",
                        devices=1,
                        max_epochs=wandb_config["epochs"],
                        min_epochs=wandb_config["epochs"],
                        logger=wandb_logger,
                        gradient_clip_val=wandb_config["gradient_clipping"],
                        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min'), 
                                checkpoint_callback])
  
    trainer.fit(module, train_dataloader, val_dataloader)

    #save model
    torch.save(module.state_dict(), "model.pt")
    


def train_sweep():

    # sweep config
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'values': [1e-7, 3e-7, 5e-7, 1e-6, 3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4]
            },
            'batch_size': {
                'values': [config.batch_size]
            },
            'epochs': {
                'values': [10]
            },
            'gradient_clipping': {
                'values': [1.0, 2.0]
            },
            'sentence_model_name': {
                'values': ["bert-base-cased", "roberta-base"]
            },
            'gloss_model_name': {
                'values': ["bert-base-cased", "roberta-base"]
            },
            'max_len': {
                'values': [config.max_len]
            },
            'loss': {
                'values': ["cross_entropy"]
            },
            'weight_decay': {
                'values': [0, 1e-5, 1e-6, 1e-7, 1e-8]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="nlp2023-hw2")
    wandb.agent(sweep_id, sweep_model)


def train():
    sentence_model_tokenizer = AutoTokenizer.from_pretrained(config.sentence_model_name)
    gloss_model_tokenizer = AutoTokenizer.from_pretrained(config.gloss_model_name)
    dataLoader = WSDDatasetLoader(config.coarse_path, config.fine_path, config.mapping_path, sentence_model_tokenizer, gloss_model_tokenizer, config.max_len)
    train_dataset, val_dataset, test_dataset = dataLoader.get_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    hyperparameters = {
        'gloss_model_name': config.gloss_model_name,
        'sentence_model_name': config.sentence_model_name,
        'lr': config.lr,
        'batch_size': config.batch_size,
        'gradient_clipping': config.gradient_clipping,
        'loss': 'cross_entropy',
        'weight_decay': config.weight_decay
    }

    # load model
    model = BiEncoder(hyperparameters)

    # logger
    # wandb_logger = WandbLogger(project="nlp2023-hw2")

    # checkpointer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='model',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    # train

    trainer = pl.Trainer(accelerator="cuda",
                         devices=1,
                         max_epochs=config.epochs,
                         min_epochs=config.epochs,
                        #  logger=wandb_logger,
                         gradient_clip_val=config.gradient_clipping,
                         callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min'), 
                                    checkpoint_callback])
    
    trainer.fit(model, train_dataloader, val_dataloader)