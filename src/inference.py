import torch
from collaction import create_text_transfrom
from data_preprocess import Data_preprocess
from train import greedy_decode

text_transfrom = create_text_transfrom()

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transfrom[Data_preprocess.SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=Data_preprocess.BOS_IDX).flatten()
    return " ".join(Data_preprocess.vocab_transform[Data_preprocess.TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")