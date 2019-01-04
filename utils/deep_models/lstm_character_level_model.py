from torch import nn
import torch
from torch.nn import MaxPool2d

from utils.params import ChrLevelLSTMParams


# input:  character embed matrix representing single word
# flow:
# - Embedding layer (char-level)
# - LSTM
# - Max-pool
class CharacterLSTMEmbed(nn.Module):
    def __init__(self, params: ChrLevelLSTMParams):
        super(CharacterLSTMEmbed, self).__init__()
        # Embedding layer
        self._embeddings = nn.Embedding(params.EMBED_vocab_dim, params.EMBED_dim)
        self._lstm = nn.LSTM(params.EMBED_dim, params.LSTM_hidden_dim, params.LSTM_layers, dropout=params.LSTM_dropout,
                             batch_first=True)

    def forward(self, x):
        x = self._embeddings(x)
        mp1 = MaxPool2d((x.shape[2], 1))
        x = torch.stack([mp1(self._lstm(x[idx, :])[0]).squeeze(dim=1) for idx in range(x.shape[0])])
        return x


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, TRAIN_ANNOTATION_SRC
    from utils.data_preprocess.corpus_loader import CorpusLoader
    from utils.data_preprocess.glove_loader import GloVeLoader
    from utils.data_preprocess.annotation_loader import AnnotationLoader
    from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
    from torch.utils.data import DataLoader
    corpus = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    labels = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))
    model = CharacterLSTMEmbed(ChrLevelLSTMParams())

    ds = RelationExtractionDataset(corpus, glove, labels=labels)
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (sent_id_, per_, org_, sent_, label) in enumerate(dl):
        per_words, per_idx = per_
        org_words, org_idx = org_
        sent_words, chr_embed, word_embed, pos_embed, ner_embed, tree_parent = sent_
        out = model(chr_embed)
        e = 0
