import torch
from timeit import default_timer as timer
from define_training_function import train_epoch, evaluate
from parameters import hyper_parameter
from model import generate_square_subsequent_mask
from data_preprocess import Data_preprocess

NUM_EPOCHS = 18
devices = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(devices)

param = hyper_parameter()
param.define()

data = Data_preprocess()
data.load_data()

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(param.transformer, param.optimizer)
    end_time = timer()
    val_loss = evaluate(param.transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == data.EOS_IDX:
            break
    return ys
