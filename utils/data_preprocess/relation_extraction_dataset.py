import torch
from collections import Counter
import en_core_web_sm
from torch.utils.data import Dataset
import numpy as np
from utils.data_preprocess.vocabulary import CharVocab, CostumeVocabulary, VocabularyBase
from utils.params import NER_OTHER, PER, ORG, UNKNOWN, PAD, UPPER

"""
This class will get sentences + labels as dict.
 - sentences format:
    {0: "I am working for the BIU-university and Karin wrote a code for Microsoft", 1: "Joe is a cashier at AM-PM"]
 - labels format(default=None):
    [ 0: [("Oved", "the BIU-university"), ("Karin", "Microsoft")],     1: [("Joe", "AM-PM")] ]

The class will provide semantic and linguistic parsing of the sentence,
and it will crete infrastructure for Embedding layers for later deep learning.
the class will provide:
 - NER tag embeddings - limited for (PER / ORG / OTHER)
 - POS tag embeddings 
 - Word Embeddings 
 - Char-Level-Embeddings 

the get_item iterator will provide examples in the following format 
 - <NP-Chunks-PER>, <NP-Chunks-ORG>, <sentence>

 each NP-Chunk is given as:
    tuple of  <list_of_char_embeddings>, <word_embeddings> 
 the sentence is given as a list of word_level_vectors
 (<list_of_char_embeddings>, <word_embeddings>, <NER embeddings>, <POS embeddings>), index_parent
"""


class RelationExtractionDataset(Dataset):
    # input:
    # - samples:    dictionary of sentences
    # - labels:     dictionary of corresponding labels for each sentence, (list of Relation tuples)
    # - vocab:      Vocabulary object
    def __init__(self, samples: dict, word_vocab: VocabularyBase, chr_vocab: CharVocab=None,
                 pos_vocab: VocabularyBase=None, ner_vocab: VocabularyBase=None, labels: dict=None):
        self._per_org = [PER, ORG]
        self._nlp = en_core_web_sm.load()
        # create embedding vocabularies for Characters, Words, POS and NER
        self._word_vocab = word_vocab
        self._chr_vocab = CharVocab(dim=2) if chr_vocab is None else chr_vocab
        self._pos_vocab = CostumeVocabulary(list(self._nlp.vocab.morphology.tag_map.keys()), dim=2) \
            if pos_vocab is None else chr_vocab
        self._ner_vocab = CostumeVocabulary([NER_OTHER, PER, ORG, UPPER], dim=2) if ner_vocab is None else chr_vocab

        # process data
        self._labels = labels
        self._sentences_rep, self._samples = self._process_samples(samples)

    def label(self, index):
        return self._samples[index][3]

    @property
    def label_count(self):
        return Counter([x[3] for x in self._samples])

    @property
    def word_vocab(self):
        return self._word_vocab

    @property
    def chr_vocab(self):
        return self._chr_vocab

    @property
    def pos_vocab(self):
        return self._pos_vocab

    @property
    def ner_vocab(self):
        return self._ner_vocab

    def __len__(self):
        return len(self._samples)

    def _get_word_embed(self, word, lemma):
        # try in that order -> WORD, LEMMA, UNKNOWN
        word_tran = self._word_vocab.translate(word.lower())
        lemma_tran = self._word_vocab.translate(lemma.lower())
        if word_tran != -1:     # try word embed
            return word_tran
        elif lemma_tran != -1:  # try lemma embed
            return lemma_tran
        else:                   # give_up -> unknown
            return self._word_vocab.translate(UNKNOWN)

    def get_ner_score(self, np_chunk):
        scores = np.zeros((len(np_chunk), self._nlp.entity.model.nr_class))
        with self._nlp.entity.step_through(np_chunk) as state:
            while not state.is_final:
                action = state.predict()
                next_tokens = state.queue
                scores[next_tokens[0].i] = state.scores
                state.transition(action)
        return scores

    def _np_chunk_combination(self, sent):
        person_np, org_np, curr_per, curr_org = [], [], [], []
        state = sent[0].ent_type_
        for i, word in enumerate(sent):
            word_ner = word.ent_type_
            if state != word_ner:
                if state == PER:
                    person_np.append(sent[curr_per[0]:curr_per[-1] + 1])
                    curr_per = []
                elif state == ORG:
                    org_np.append(sent[curr_org[0]:curr_org[-1] + 1])
                    curr_org = []
                state = word_ner
            if state == PER:
                curr_per.append(i)
            elif state == ORG:
                curr_org.append(i)

        # loop on all options person-org
        for per in person_np:
            for org in org_np:
                yield per, org

    def _get_char_embed(self, word):
        word_chr_embed = []  # char embeddings
        for c in word:
            word_chr_embed.append(self._chr_vocab.translate(c))
        return word_chr_embed

    def _get_np_root(self, np_phrase):
        for w in self._nlp(np_phrase):
            if w.dep_ == "ROOT":
                return w.text

    # because annotations gold label and np chunks can be slightly different
    # e.g gold value is: "Air Line Pilots Association" and the np chunk value is "The Air Line Pilots Association"
    # we tell if there are referring to the same object by looking at the ROOT word at the NP-chunk/Gold-val
    def _get_label(self, sent_id, per, org):
        label = 0
        if self._labels:
            org_root = self._get_np_root(org.text)
            # loop over list of [ .. (per, org, org_root) .. ]
            for sent_per, sent_org, sent_org_root in self._labels.get(sent_id, []):
                # check similarity between PERSON and ORGANIZATION-ROOT
                if self._get_np_root(per.text) == self._get_np_root(sent_per) and org_root == sent_org_root:
                    label = 1
                    break
        return label

    def _process_samples(self, samples: dict):
        sent_rep = {}
        data = []
        for sent_id, sent_str in samples.items():
            sent = self._nlp(sent_str)
            # word level features
            words = []
            chr_embed = []
            word_embed = []
            ner_embed = []
            pos_embed = []
            tree_parent = []
            for i, word in enumerate(sent):
                words.append(word.text)                                            # actual text
                tree_parent.append(0 if word.dep_ == "ROOT" else word.head.i - i)  # tree
                chr_embed.append(self._get_char_embed(word.text))                  # chr embed
                word_embed.append(self._get_word_embed(word.text, word.lemma_))    # word embed
                pos_embed.append(self._pos_vocab.translate(word.tag_))             # pos embed
                word_ent = word.ent_type_                                          # ner embed
                ner_embed.append(self._ner_vocab.translate(word_ent) if word_ent in self._per_org
                                 else (self._ner_vocab.translate(NER_OTHER) if word.text[0].isupper() else
                                       self._ner_vocab.translate(NER_OTHER)))

            # save representation by sentence-ID
            sent_rep[sent_id] = (words, chr_embed, word_embed, pos_embed, ner_embed, tree_parent)
            # for each combination of (person, org)
            # append (person, org, sentence-ID) triplet to the data
            for per, org in self._np_chunk_combination(sent):
                data.append((per, org, sent_id, self._get_label(sent_id, per, org)))
        return sent_rep, data

    @property
    def semantic_samples(self):
        for i in range(len(self._samples)):
            per, org, sent_id, label = self._samples[i]
            words, chr_embed, word_embed, pos_embed, ner_embed, tree_parent = self._sentences_rep[sent_id]
            tree_parent = [val + i for i, val in enumerate(tree_parent)]
            per_idx = [w.i for w in per]
            org_idx = [w.i for w in org]
            yield sent_id, (per, per_idx), (org, org_idx), (words, tree_parent), label

    def __getitem__(self, index):
        per, org, sent_id, label = self._samples[index]
        seq_rep = self._sentences_rep[sent_id]
        # get indexes in sentence and convert to txt list
        per_idx = [w.i for w in per]
        org_idx = [w.i for w in org]
        per = per.text.split()
        org = org.text.split()

        # word level embeddings for person and org
        per_word_rep = []
        for w in per:
            per_word_rep.append(self._get_word_embed(w, w))
        org_word_rep = []
        for w in org:
            org_word_rep.append(self._get_word_embed(w, w))
        return sent_id, (per, per_idx), (org, org_idx), seq_rep, label

    def collate_fn(self, batch):
        lengths_sent = []
        lengths_chr_sent = []

        # calculate max word len + max char len
        for sample in batch:
            sent_id, per, org, seq_rep, label = sample
            words, chr_embed, word_embed, pos_embed, ner_embed, tree_parent = seq_rep

            lengths_sent.append(len(words))
            lengths_chr_sent.append(len(max(words, key=lambda x: len(x))))

        # in order to pad all batch to a single dimension max length is needed
        max_sent = np.max(lengths_sent)
        max_chr_sent = np.max(lengths_chr_sent)

        # new batch variables
        per_text_batch = []
        per_idx_batch = []

        org_text_batch = []
        org_idx_batch = []

        # new batch variables
        sent_id_batch = []
        sent_text_batch = []
        sent_chr_batch = []
        sent_words_batch = []
        sent_pos_batch = []
        sent_ner_batch = []
        tree_parent_batch = []

        # labels
        labels = []

        for sample in batch:
            sent_id, per, org, seq_rep, label = sample
            per_words, per_idx = per
            org_words, org_idx = org
            sent_words, chr_embed, word_embed, pos_embed, ner_embed, tree_parent = seq_rep

            labels.append(label)
            # original sentences as lists no need to pad
            sent_id_batch.append(sent_id)
            per_text_batch.append(per_words)
            org_text_batch.append(org_words)
            sent_text_batch.append(sent_words)
            # pad word level embedding vectors
            per_idx_batch.append([1 if idx in per_idx else 0 for idx in range(max_sent)])
            org_idx_batch.append([1 if idx in org_idx else 0 for idx in range(max_sent)])
            sent_words_batch.append(word_embed + [self._word_vocab.translate(PAD)] * (max_sent - len(word_embed)))
            sent_pos_batch.append(pos_embed + [self._word_vocab.translate(PAD)] * (max_sent - len(pos_embed)))
            sent_ner_batch.append(ner_embed + [self._word_vocab.translate(PAD)] * (max_sent - len(ner_embed)))
            tree_parent_batch.append(tree_parent + [0] * (max_sent - len(tree_parent)))
            # pad char vector at char level and at word level
            temp = [[self._chr_vocab.translate(PAD)] * (max_chr_sent - len(chars)) + chars for chars in chr_embed]
            sent_chr_batch.append([[self._chr_vocab.translate(PAD)] * max_chr_sent] * (max_sent - len(chr_embed)) + temp)
        return sent_id_batch, (per_text_batch, torch.Tensor(per_idx_batch).long()), \
            (org_text_batch, torch.Tensor(org_idx_batch).long()), \
            (sent_text_batch, torch.Tensor(sent_chr_batch).long(), torch.Tensor(sent_words_batch).long(),
             torch.Tensor(sent_pos_batch).long(), torch.Tensor(sent_ner_batch).long(),
             torch.Tensor(tree_parent_batch)), torch.Tensor(labels).long()


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, TRAIN_ANNOTATION_SRC
    from utils.data_preprocess.corpus_loader import CorpusLoader
    from utils.data_preprocess.glove_loader import GloVeLoader
    from utils.data_preprocess.annotation_loader import AnnotationLoader
    from torch.utils.data import DataLoader
    corpus = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    labels = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))

    ds = RelationExtractionDataset(corpus, glove, labels=labels)
    dl = DataLoader(
        dataset=ds,
        batch_size=3,
        collate_fn=ds.collate_fn
    )
    for i, (sent_id_, per_, org_, sent_, label_) in enumerate(dl):
        print(per_, org_, sent_, label_)
