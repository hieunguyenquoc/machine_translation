import torch
from torch.nn.utils.rnn import pad_sequence
from data_preprocess import Data_preprocess
from typing import List

'''Convert strings pairs (in two language) into the batched tensors
   Basically convert string to tensor so pytorch can handle
'''
data = Data_preprocess()
data.load_data()

class Collaction:
# helper function to club together sequential operations
    def __init__(self) :
        self.text_transform = {}
        for ln in [data.SRC_LANGUAGE, data.TGT_LANGUAGE]:
            self.text_transform[ln] = self.sequential_transforms(data.token_transform[ln], #Tokenization
                                                    data.vocab_transform[ln], #Numericalization
                                                    self.tensor_transform) # Add BOS/EOS and create tensor
            
    def sequential_transforms(*transforms): # * : function takes a variable number arguments
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(token_ids: List[int]):
        return torch.cat((torch.tensor([data.SRC_LANGUAGE]), #.cat : concatenates multiple into one tensor
                        torch.tensor(token_ids),
                        torch.tensor([data.TGT_LANGUAGE])))

    # src and tgt language text transforms to convert raw strings into tensors indices
    
    # function to collate data samples into batch tesors
    def collate_fn(self,batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[data.SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[data.TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=data.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=data.PAD_IDX)
        return src_batch, tgt_batch