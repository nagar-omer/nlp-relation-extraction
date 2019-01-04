from torch.nn import MaxPool1d, Module, Embedding, LSTM, AvgPool1d, Dropout
from utils.params import EncoderDecoderParams
import torch


class EncoderDecoder(Module):
    def __init__(self, params: EncoderDecoderParams):
        super(EncoderDecoder, self).__init__()
        # word embed layer
        self._embeddings_words = self._load_pre_trained(params.EMBED_WORDS_pre_trained, params.GPU) if \
            params.EMBED_WORDS_use_pre_trained else Embedding(params.EMBED_WORDS_vocab_dim, params.EMBED_WORDS_dim)
        self._embeddings_pos = Embedding(params.EMBED_POS_vocab_dim, params.EMBED_POS_dim)
        self._embeddings_ner = Embedding(params.EMBED_NER_vocab_dim, params.EMBED_NER_dim)
        self._dropout = Dropout(p=params.LSTM_dropout_0)

        # ========================================= ENCODER =========================================
        self._encoder_layer_0 = LSTM(1 + params.EMBED_WORDS_dim + params.EMBED_CHR_dim + params.EMBED_POS_dim +
                                     params.EMBED_NER_dim, params.LSTM_hidden_dim, num_layers=1,
                                     batch_first=True, bidirectional=True)
        self._encoder_layer_1 = LSTM(1 + params.EMBED_WORDS_dim + params.EMBED_CHR_dim + (2 * params.LSTM_hidden_dim),
                                     params.LSTM_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self._encoder_layer_2 = LSTM(1 + params.EMBED_WORDS_dim + params.EMBED_CHR_dim + (2 * params.LSTM_hidden_dim),
                                     params.LSTM_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        # ========================================= DECODER =========================================
        self._decoder_layer_0 = LSTM(params.EMBED_WORDS_dim + params.EMBED_CHR_dim + (8 * params.LSTM_hidden_dim),
                                     params.LSTM_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self._decoder_layer_1 = LSTM(params.EMBED_WORDS_dim + params.EMBED_CHR_dim + (2 * params.LSTM_hidden_dim),
                                     params.LSTM_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self._decoder_layer_2 = LSTM(params.EMBED_WORDS_dim + params.EMBED_CHR_dim + (2 * params.LSTM_hidden_dim),
                                     params.LSTM_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    @staticmethod
    def _load_pre_trained(weights_matrix, gpu, non_trainable=False):
        weights_matrix = torch.Tensor(weights_matrix).cuda() if gpu else torch.Tensor(weights_matrix).cuda()
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    # implement attention  Main paper -- 3.3 Composition Layer --
    def _calc_attention_coefficients(self):
        # get LSTM gate parameters
        # w_ii, w_if, w_ic, w_io = self._lstm_layer.weight_ih_l0.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = self._encoder_layer_2.weight_hh_l0.chunk(4, 0)
        reverse_w_hi, w_hf, w_hc, reverse_w_ho = self._encoder_layer_2.weight_hh_l0_reverse.chunk(4, 0)
        norm_out_gates = torch.norm(torch.cat([w_hi, reverse_w_hi], dim=0), dim=1)
        attention_coefficient_i = norm_out_gates / torch.sum(norm_out_gates)

        norm_out_gates = torch.norm(torch.cat([w_ho, reverse_w_ho], dim=0), dim=1)
        attention_coefficient_o = norm_out_gates / torch.sum(norm_out_gates)
        return attention_coefficient_i, attention_coefficient_o

    def forward(self, words_embed, pos_embed, ner_embed, chr_rep, semantic_tree, per_mask, org_mask):
        attention_coefficients_i, attention_coefficients_o = self._calc_attention_coefficients()
        # dynamic average and max pool according to batch sentence length
        activate_avg_pool = AvgPool1d(words_embed.shape[1], 1)
        activate_max_pool = MaxPool1d(words_embed.shape[1], 1)

        # concat following [ semantic_parent | chr_embed | word_embed | POS_embed | NER_embed ]
        semantic_tree = semantic_tree.unsqueeze(dim=2)
        embed_word = self._embeddings_words(words_embed)
        x = torch.cat([semantic_tree, chr_rep, embed_word, self._embeddings_pos(pos_embed),
                       self._embeddings_ner(ner_embed)], dim=2)

        # ========================================= ENCODER =========================================
        # 3 layers Bi-LSTM + skip connections + dropout layers in between
        output_seq, _ = self._encoder_layer_0(x)
        output_seq = self._dropout(output_seq)
        output_seq, _ = self._encoder_layer_1(torch.cat([semantic_tree, chr_rep, embed_word, output_seq], dim=2))
        output_seq = self._dropout(output_seq)
        output_seq, _ = self._encoder_layer_2(torch.cat([semantic_tree, chr_rep, embed_word, output_seq], dim=2))

        # attention
        avg_pool = activate_avg_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        max_pool = activate_max_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        gate_attention_i = torch.sum(output_seq * attention_coefficients_i, dim=1)
        gate_attention_o = torch.sum(output_seq * attention_coefficients_o, dim=1)
        x = torch.cat([gate_attention_i, gate_attention_o, avg_pool, max_pool], dim=1).unsqueeze(dim=1)
        x = torch.stack([x.squeeze(dim=1) for _ in range(embed_word.shape[1])]).transpose(0, 1)

        # ========================================= DECODER =========================================
        output_seq, _ = self._decoder_layer_0(torch.cat([chr_rep, embed_word, x], dim=2))
        output_seq = self._dropout(output_seq)
        output_seq, _ = self._decoder_layer_1(torch.cat([chr_rep, embed_word, output_seq], dim=2))
        output_seq = self._dropout(output_seq)
        output_seq, _ = self._decoder_layer_2(torch.cat([chr_rep, embed_word, output_seq], dim=2))

        per_mask = torch.stack([per_mask for _ in range(output_seq.shape[2])], dim=2).float()
        org_mask = torch.stack([org_mask for _ in range(output_seq.shape[2])], dim=2).float()

        max_pool = MaxPool1d(output_seq.shape[1])
        per_org_rep = torch.cat([max_pool((per_mask*output_seq).transpose(1,2)).squeeze(dim=2),
                                 max_pool((org_mask * output_seq).transpose(1, 2)).squeeze(dim=2)], dim=1)
        # per_org_rep = torch.cat([torch.sum((org_mask*output_seq), dim=1),
        #                          torch.sum((per_mask*output_seq), dim=1)], dim=1)
        return per_org_rep


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, TRAIN_ANNOTATION_SRC, ChrLevelLSTMParams
    from utils.data_preprocess.corpus_loader import CorpusLoader
    from utils.data_preprocess.glove_loader import GloVeLoader
    from utils.data_preprocess.annotation_loader import AnnotationLoader
    from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
    from utils.deep_models.lstm_character_level_model import CharacterLSTMEmbed
    from torch.utils.data import DataLoader

    corpus = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    labels = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))
    ds = RelationExtractionDataset(corpus, glove, labels=labels)
    chr_model_params = ChrLevelLSTMParams()
    att_params = EncoderDecoderParams(len(ds.word_vocab), len(ds.pos_vocab), len(ds.ner_vocab),
                                      chr_model_params.OUT_DIM, pre_trained=ds.word_vocab.embeddings_mx)
    chr_model = CharacterLSTMEmbed(chr_model_params)
    att_model = EncoderDecoder(att_params)

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
        e = 0
