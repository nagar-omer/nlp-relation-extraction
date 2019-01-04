from torch.autograd import Variable
from torch.nn import Module
from utils.deep_models.encoder_decoder_attention_model import EncoderDecoder
from utils.deep_models.top_level_model import TopLayerModel
from utils.params import REFullModelParams
from utils.deep_models.lstm_character_level_model import CharacterLSTMEmbed


class REModel(Module):
    def __init__(self, params: REFullModelParams):
        super(REModel, self).__init__()
        self._gpu = params.GPU
        self._chr_embed_model = CharacterLSTMEmbed(params.CHARACTER_params)             # character embed (LSTM)
        self._attention_model = EncoderDecoder(params.ATTENTION_params)                 # sentence to vec (Bi-LSTM)
        self._top_layer_model = TopLayerModel(params.TOP_LAYAER_params)                 # combine all (NN)
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER, params.WEIGHT_DECAY)

    # init optimizer with RMS_prop
    def set_optimizer(self, lr, opt, wd):
        return opt(self.parameters(), lr=lr, weight_decay=wd)

    def _var(self, list_vars, grad):
        return [Variable(var).cuda() if self._gpu else Variable(var) for var in list_vars] if grad else \
            [var.cuda() if self._gpu else var for var in list_vars]

    def forward(self, per, org, sent, grad=True):
        # get variables
        per_words, per_mask = per
        org_words, org_mask = org
        per_mask, org_mask = self._var([per_mask, org_mask], grad)
        sent_words, chr_embed, word_embed, pos_embed, ner_embed, tree_parent = sent
        chr_embed, word_embed, pos_embed, ner_embed, tree_parent = \
            self._var([chr_embed, word_embed, pos_embed, ner_embed, tree_parent], grad)

        # sentence vector
        chr_rep = self._chr_embed_model(chr_embed)
        org_per_rep = self._attention_model(word_embed, pos_embed, ner_embed, chr_rep, tree_parent, per_mask, org_mask)
        return self._top_layer_model(org_per_rep)


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, TRAIN_ANNOTATION_SRC, ChrLevelLSTMParams, EncoderDecoderParams\
        , TopLayerParams
    from utils.data_preprocess.corpus_loader import CorpusLoader
    from utils.data_preprocess.glove_loader import GloVeLoader
    from utils.data_preprocess.annotation_loader import AnnotationLoader
    from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
    from torch.utils.data import DataLoader

    corpus = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    labels = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))
    ds = RelationExtractionDataset(corpus, glove, labels=labels)
    chr_model_params = ChrLevelLSTMParams()
    att_params = EncoderDecoderParams(len(ds.word_vocab), len(ds.pos_vocab), len(ds.ner_vocab),
                                      chr_model_params.OUT_DIM, pre_trained=ds.word_vocab.embeddings_mx)
    top_mlp_params = TopLayerParams(att_params.OUT_DIM)
    re_params = REFullModelParams(chr_model_params, att_params, top_mlp_params)

    chr_model = CharacterLSTMEmbed(chr_model_params)
    att_model = EncoderDecoder(att_params)
    top_mlp = TopLayerModel(top_mlp_params)
    re_model = REModel(re_params)
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (sent_id_, per_, org_, sent_, label) in enumerate(dl):
        per_words, per_mask_ = per_
        org_words, org_mask_ = org_
        sent_words, chr_embed, word_embed_, pos_embed_, ner_embed_, tree_parent_ = sent_
        out = re_model(per_, org_, sent_)
        e = 0
