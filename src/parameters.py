import torch
import torch.nn as nn
from data_preprocess import Data_preprocess
from model import Seq2SeqTransformer

torch.manual_seed(0)

class hyper_parameter:
    def __init__(self):
        devices = "cuda" if torch.cuda.is_available() else "cpu"

        DEVICE = torch.device(devices)

        self.SRC_VOCAB_SIZE = len(Data_preprocess.vocab_transform[Data_preprocess.SRC_LANGUAGE])
        self.TGT_VOCAB_SIZE = len(Data_preprocess.vocab_transform[Data_preprocess.TGT_LANGUAGE])
        self.EMB_SIZE = 512
        self.NHEAD = 8
        self.FFN_HID_DIM = 512
        self.BATCH_SIZE = 128
        self.NUM_ENCODER_LAYERS = 3
        self.NUM_DECODER_LAYERS = 3

        self.transformer = Seq2SeqTransformer(self.NUM_ENCODER_LAYERS, self.NUM_DECODER_LAYERS, self.EMB_SIZE,
                                        self.NHEAD, self.SRC_VOCAB_SIZE, self.TGT_VOCAB_SIZE, self.FFN_HID_DIM)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.transformer = self.transformer.to(DEVICE)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=Data_preprocess.PAD_IDX)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)