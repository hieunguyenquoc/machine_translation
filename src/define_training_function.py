import torch
from torch.utils.data import DataLoader
from torchtext.datasets import  Multi30k
from data_preprocess import Data_preprocess
from parameters import hyper_parameter
from model import create_mask
from collaction import collate_fn

devices = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = torch.device(devices)

result = Data_preprocess()
result.load_data()

hyper_param = hyper_parameter()
hyper_param.define()

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(result.SRC_LANGUAGE, result.TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=hyper_param.BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input) #input of model

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask) 

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = hyper_parameter.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(result.SRC_LANGUAGE, result.TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=hyper_param.BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = hyper_param.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)