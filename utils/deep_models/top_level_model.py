from torch.nn import Module, Linear
from torch.nn.functional import softmax
from utils.params import TopLayerParams
import torch


class TopLayerModel(Module):
    def __init__(self, params: TopLayerParams):
        super(TopLayerModel, self).__init__()
        # useful info in forward function
        self._layer0 = Linear(params.LINEAR_in_dim, params.LINEAR_hidden_dim_0)
        self._layer1 = Linear(params.LINEAR_hidden_dim_0, params.LINEAR_hidden_dim_1)
        self._output_layer = Linear(params.LINEAR_hidden_dim_1, params.LINEAR_out_dim)
        self._activation = params.Activation

    def forward(self, x):
        x = self._layer0(x)
        x = self._activation(x)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._output_layer(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, TRAIN_ANNOTATION_SRC, ChrLevelLSTMParams, EncoderDecoderParams
    from utils.data_preprocess.corpus_loader import CorpusLoader
    from utils.data_preprocess.glove_loader import GloVeLoader
    from utils.data_preprocess.annotation_loader import AnnotationLoader
    from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
    from utils.deep_models.lstm_character_level_model import CharacterLSTMEmbed
    from torch.utils.data import DataLoader
    from utils.deep_models.encoder_decoder_attention_model import EncoderDecoder

    corpus = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    labels = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))
    ds = RelationExtractionDataset(corpus, glove, labels=labels)
    chr_model_params = ChrLevelLSTMParams()
    att_params = EncoderDecoderParams(len(ds.word_vocab), len(ds.pos_vocab), len(ds.ner_vocab),
                                      chr_model_params.OUT_DIM, pre_trained=ds.word_vocab.embeddings_mx)
    top_mlp_params = TopLayerParams(att_params.OUT_DIM)
    chr_model = CharacterLSTMEmbed(chr_model_params)
    att_model = EncoderDecoder(att_params)
    top_mlp = TopLayerModel(top_mlp_params)

    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (sent_id_, per_, org_, sent_, label) in enumerate(dl):
        per_words, per_mask_ = per_
        org_words, org_mask_ = org_
        sent_words, chr_embed, word_embed_, pos_embed_, ner_embed_, tree_parent_ = sent_
        chr_rep_ = chr_model(chr_embed)
        out = att_model(word_embed_, pos_embed_, ner_embed_, chr_rep_, tree_parent_, per_mask_, org_mask_)
        out = top_mlp(out)
        e = 0
