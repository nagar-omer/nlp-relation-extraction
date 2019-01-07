import os
from torch.nn.functional import relu, cross_entropy, nll_loss, binary_cross_entropy, tanh
from torch.optim import Adam, SGD, RMSprop
import torch

# ------------------------------ Data params ------------------------------
RELATION = "Work_For"

PER = "PERSON"
ORG = "ORG"
UPPER = "UPPER"
NER_OTHER = "O"

PRE_TRAINED_SRC = os.path.join("data", "GloVe", "glove.6B.50d.txt")
TRAIN_SRC = os.path.join("data", "RE", "Corpus.TRAIN.txt")
DEV_SRC = os.path.join("data", "RE", "Corpus.DEV.txt")
TRAIN_ANNOTATION_SRC = os.path.join("data", "RE", "TRAIN.annotations")
DEV_ANNOTATION_SRC = os.path.join("data", "RE", "DEV.annotations")

UNKNOWN = "UUNNKK"          # token for unknown words
PAD = "<p>"                 # token for artificial padding


# ------------------------------ Models params ------------------------------
class ChrLevelLSTMParams:
    def __init__(self, chr_vocab_dim=129):
        self.EMBED_dim = 50
        self.EMBED_vocab_dim = chr_vocab_dim
        self.LSTM_hidden_dim = 50
        self.LSTM_layers = 3
        self.LSTM_dropout = 0.3
        self.OUT_DIM = self.LSTM_hidden_dim


class EncoderDecoderParams:
    def __init__(self, word_vocab_dim, pos_vocab_dim, ner_vocab_dim, chr_rep_dim, pre_trained=None, gpu=True):
        self.ATTENTION_activation = relu
        self.EMBED_WORDS_pre_trained = pre_trained
        self.EMBED_WORDS_use_pre_trained = True if pre_trained is not None else False
        self.EMBED_WORDS_vocab_dim = word_vocab_dim
        self.EMBED_WORDS_dim = 50
        self.EMBED_POS_vocab_dim = pos_vocab_dim
        self.EMBED_POS_dim = 30
        self.EMBED_NER_vocab_dim = ner_vocab_dim
        self.EMBED_NER_dim = 3
        self.EMBED_CHR_dim = chr_rep_dim                # number of filters at chr level model
        self.LSTM_layers = 1
        self.LSTM_hidden_dim = 200
        self.LSTM_dropout_0 = 0.15
        self.LSTM_dropout_1 = 0.15
        self.LSTM_dropout_2 = 0
        self.GPU = gpu
        self.OUT_DIM = self.LSTM_hidden_dim * 4
        # OUT_DIM = (batch_size, max_sent_len, self.LSTM_hidden_dim * 2)


class TopLayerParams:
    def __init__(self, in_dim):
        self.LINEAR_in_dim = in_dim                     # should be 4 * SequenceEncoderParams::OUT_DIM
        self.LINEAR_hidden_dim_0 = int(in_dim * 0.8)
        self.LINEAR_hidden_dim_1 = int(in_dim * 0.4)
        self.LINEAR_out_dim = 1
        self.Activation = torch.tanh
        self.OUT_DIM = 1


class REFullModelParams:
    def __init__(self, chr_params, attention_params, top_layer_params, gpu=True):
        self.CHARACTER_params = chr_params
        self.ATTENTION_params = attention_params
        self.TOP_LAYAER_params = top_layer_params
        self.LEARNING_RATE = 1e-4
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 1e-2
        self.GPU = gpu


# ----------------------------- Activator Params -----------------------------
class REActivatorParams:
    def __init__(self):
        self.LOSS = binary_cross_entropy
        self.BATCH_SIZE = 32
        self.GPU = True
        self.EPOCHS = 30
        self.VALIDATION_RATE = 10
