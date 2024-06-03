import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, concatenate, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, concatenate, Conv1D, MaxPooling1D, Flatten, Bidirectional, GlobalMaxPooling1D, Concatenate
from keras.models import Model
import torch
import argparse
from fairseq.data import Dictionary
import pickle


from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
# Define and parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes',
    default="PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()

MAX_LEN = 256
# Define the architecture of your new model
vocab = Dictionary()
vocab.add_from_file("PhoBERT_base_transformers/dict.txt")

# Load BPE codes
bpe = fastBPE(args)

def make_mask(batch_ids):
    batch_mask = []
    for ids in batch_ids:
        mask = [int(token_id > 0) for token_id in ids]
        batch_mask.append(mask)
    return torch.tensor(batch_mask)

def preprocess_single_sentence(title, text):
    # Áp dụng các bước tiền xử lý và mã hóa dữ liệu cho title
    title = str(title)
    title_subwords = '<s> ' + bpe.encode(title) + ' </s>'
    title_encoded = vocab.encode_line(title_subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    title_ids = pad_sequences([title_encoded], maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    title_mask = make_mask(title_ids)
    tensor_title_ids = torch.tensor(title_ids)
    tensor_title_mask = torch.tensor(title_mask)

    # Áp dụng các bước tiền xử lý và mã hóa dữ liệu cho text
    text = str(text)
    text_subwords = '<s> ' + bpe.encode(text) + ' </s>'
    text_encoded = vocab.encode_line(text_subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    text_ids = pad_sequences([text_encoded], maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    text_mask = make_mask(text_ids)
    tensor_text_ids = torch.tensor(text_ids)
    tensor_text_mask = torch.tensor(text_mask)

    # Trả về tensor đại diện cho title và text, cùng các mask tương ứng
    return tensor_title_ids, tensor_text_ids

# Lưu biến vocab vào file
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

