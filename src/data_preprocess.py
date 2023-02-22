from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List

class Data_preprocess:
# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
    def load_data(self):
        multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
        multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

        self.SRC_LANGUAGE = 'de'
        self.TGT_LANGUAGE = 'en'

        # Place-holders
        self.token_transform = {}
        self.vocab_transform = {}

        # Create source and target language tokenizer. Make sure to install the dependencies.
        # pip install -U torchdata
        # pip install -U spacy
        # python -m spacy download en_core_web_sm
        # python -m spacy download de_core_news_sm
        self.token_transform[self.SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm') #return list of tokens
        self.token_transform[self.TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm') #return list of tokens

        # Define special symbols and indices
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


    # helper function to yield list of tokens
    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        language_index = {self.SRC_LANGUAGE: 0, self.TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]]) #yield : cho phép hàm trả về nhiều giá trị

result = Data_preprocess()
result.load_data()

for ln in [result.SRC_LANGUAGE, result.TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(result.SRC_LANGUAGE, result.TGT_LANGUAGE))
    # Create torchtext's Vocab object 
    result.vocab_transform[ln] = build_vocab_from_iterator(result.yield_tokens(train_iter, ln), #vocab_transfrom[ln] : vocab tương ứng với ngôn ngữ
                                                    min_freq=1,
                                                    specials=result.special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [result.SRC_LANGUAGE, result.TGT_LANGUAGE]:
    result.vocab_transform[ln].set_default_index(result.UNK_IDX)
    