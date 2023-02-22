import torch
from torch.nn.utils.rnn import pad_sequence
from data_preprocess import Data_preprocess
from typing import List

'''Convert strings pairs (in two language) into the batched tensors
   Basically convert string to tensor so pytorch can handle
'''
# helper function to club together sequential operations
def sequential_transforms(*transforms): # * : function takes a variable number arguments
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([Data_preprocess.BOS_IDX]), #.cat : concatenates multiple into one tensor
                      torch.tensor(token_ids),
                      torch.tensor([Data_preprocess.EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
def create_text_transfrom():
    text_transform = {}
    for ln in [Data_preprocess.SRC_LANGUAGE, Data_preprocess.TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(Data_preprocess.token_transform[ln], #Tokenization
                                                Data_preprocess.vocab_transform[ln], #Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor
    return text_transform

text_transfrom = create_text_transfrom()

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transfrom[Data_preprocess.SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transfrom[Data_preprocess.TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=Data_preprocess.PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=Data_preprocess.PAD_IDX)
    return src_batch, tgt_batch