from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import transformers
from transformers import AutoModel
import pytorch_lightning as pl
import hw2.stud.config as config
import loralib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math


class GlossEncoder(torch.nn.Module):
    '''
    Encodes a gloss using a pretrained model.
    '''

    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name

    def forward(self, input_ids, attention_mask, gloss_output_mask):
        return self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :] * gloss_output_mask.unsqueeze(-1)
    
    def set_train(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
        
        for param in self.model.encoder.layer[-2].parameters():
            param.requires_grad = True
    
    def set_eval(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    
    
class ContextEncoder(torch.nn.Module):
    '''
    Encodes a context using a pretrained model.
    The forward method returns the hidden states of the requested tokens.
    '''

    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name

    def forward(self, input_ids, attention_mask, context_output_mask):
        return torch.mean(self.model(input_ids, attention_mask=attention_mask).last_hidden_state[context_output_mask.bool(), :], dim=0).unsqueeze(0)
    
    def set_train(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
        
        for param in self.model.encoder.layer[-2].parameters():
            param.requires_grad = True
    
    def set_eval(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
class BiEncoder(pl.LightningModule):

    def __init__(self, hyperparameters):
        super(BiEncoder, self).__init__()
        self.hyperparameters = hyperparameters
        self.gloss_encoder = GlossEncoder(hyperparameters["gloss_model_name"])
        self.context_encoder = ContextEncoder(hyperparameters["sentence_model_name"])
        if hyperparameters["loss"] == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=config.padding_token)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.preds = []
        self.ground_truth = []
        self.val_preds = []
        self.val_ground_truth = []
        self.is_training_gloss = True
        self.cc = torch.nn.CosineSimilarity()

    def forward(self, gloss_input_ids, gloss_attention_mask, context_input_ids, context_attention_mask, context_output_mask, gloss_output_mask):
        res = []
        for i in range(len(gloss_input_ids)):
            context_hidden_states = self.context_encoder(context_input_ids[i], context_attention_mask[i], context_output_mask[i])
            gloss_hidden_states = self.gloss_encoder(gloss_input_ids[i], gloss_attention_mask[i], gloss_output_mask[i])
            
            cross_dot = self.cc(gloss_hidden_states, context_hidden_states)
            
            res.append(cross_dot)
        res = torch.stack(res)
        if len(res.shape) >= 3:
            res = res.squeeze(-1)
        return res
    
    def calculate_loss(self, cross_dot, gold_senses_idx):
        return self.loss(cross_dot, gold_senses_idx)
    
    def on_train_epoch_start(self) -> None:
        self.gloss_encoder.set_train()
        self.context_encoder.set_train()
        # if self.is_training_gloss:
        #     self.gloss_encoder.set_train()
        #     self.context_encoder.set_eval()
        # else:
        #     self.gloss_encoder.set_eval()
        #     self.context_encoder.set_train()
        # self.is_training_gloss = not self.is_training_gloss
    
    def training_step(self, batch, batch_idx):
        context_input_ids, context_attention_mask, context_output_mask, gold_sense_ids, gold_senses_idx, gloss_input_ids, gloss_attention_mask, gloss_output_mask = batch
        cross_dot = self.forward(gloss_input_ids, gloss_attention_mask, context_input_ids, context_attention_mask, context_output_mask, gloss_output_mask)
        if cross_dot.ravel()[0] == torch.nan:
            raise Exception("Nan in cross_dot")
        # add padding to the labels
        padded_gold_input_idx = torch.ones_like(cross_dot) * -100
        for idx in range(len(gold_senses_idx)):
            padded_gold_input_idx[idx, :gold_senses_idx[idx].shape[1]] = gold_senses_idx[idx][0, :]


        self.preds.extend(torch.argmax(cross_dot, dim=1).to("cpu").tolist())
        self.ground_truth.extend(torch.argmax(padded_gold_input_idx, dim=1).to("cpu").tolist())

        mask = (padded_gold_input_idx != -100)

        cross_dot = cross_dot[mask]
        padded_gold_input_idx = padded_gold_input_idx[mask]

        loss = self.calculate_loss(cross_dot.ravel(), padded_gold_input_idx.ravel())

        self.log('train_loss_batch', loss.item(), prog_bar=True, logger=True)

        self.train_step_outputs.append(loss.item())
        return loss
    

    
    def validation_step(self, batch, batch_idx):
        context_input_ids, context_attention_mask, token_idxs, gold_sense_ids, gold_senses_idx, gloss_input_ids, gloss_attention_mask, gloss_output_mask = batch
        cross_dot = self.forward(gloss_input_ids, gloss_attention_mask, context_input_ids, context_attention_mask, token_idxs, gloss_output_mask)
        # add padding to the labels
        padded_gold_input_idx = torch.ones_like(cross_dot) * -100
        for idx in range(len(gold_senses_idx)):
            padded_gold_input_idx[idx, :gold_senses_idx[idx].shape[1]] = gold_senses_idx[idx][0, :]
        

        self.val_preds.extend(torch.argmax(cross_dot, dim=1).to("cpu").tolist())
        self.val_ground_truth.extend(torch.argmax(padded_gold_input_idx, dim=1).to("cpu").tolist())


        mask = (padded_gold_input_idx != -100)

        cross_dot = cross_dot[mask]
        padded_gold_input_idx = padded_gold_input_idx[mask]

        loss = self.calculate_loss(cross_dot.ravel(), padded_gold_input_idx.ravel())

        self.log('val_loss_batch', loss.item(), prog_bar=True, logger=True)

        self.validation_step_outputs.append(loss.item())


    def on_train_epoch_end(self) -> None:
        self.log('train_loss', sum(self.train_step_outputs), prog_bar=True, logger=True)
        acc, prec, recall, f1 = self.calculate_metrics(self.preds, self.ground_truth)
        self.preds = []
        self.ground_truth = []
        self.log('train_acc', acc, prog_bar=True, logger=True)
        self.log('train_prec', prec, prog_bar=True, logger=True)
        self.log('train_recall', recall, prog_bar=True, logger=True)
        self.log('train_f1', f1, prog_bar=True, logger=True)
        self.train_step_outputs.clear()  # free mem

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', sum(self.validation_step_outputs), prog_bar=True, logger=True)
        acc, prec, recall, f1 = self.calculate_metrics(self.val_preds, self.val_ground_truth)
        self.val_preds = []
        self.val_ground_truth = []
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_prec', prec, prog_bar=True, logger=True)
        self.log('val_recall', recall, prog_bar=True, logger=True)
        self.log('val_f1', f1, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # free mem

    def calculate_metrics(self, predictions, labes):
        acc = accuracy_score(labes, predictions)
        prec = precision_score(labes, predictions, average="macro", zero_division=True)
        recall = recall_score(labes, predictions, average="macro", zero_division=True)
        f1 = f1_score(labes, predictions, average="macro", zero_division=True)
        return acc, prec, recall, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters["lr"], weight_decay=self.hyperparameters["weight_decay"])
        return optimizer
    