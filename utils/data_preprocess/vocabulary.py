"""
Interface for word embedding class
"""
from utils.params import PAD, UNKNOWN
import numpy as np


class VocabularyBase:
    @property
    # return embedding matrix
    def embeddings_mx(self):
        raise NotImplementedError()

    @property
    def embedding_dim(self):
        raise NotImplementedError()

    # return embedding matrix
    def __len__(self):
        raise NotImplementedError()

    # return index for a specific word
    def _to_idx(self, word):
        raise NotImplementedError()

    # return word for specific index
    def _from_idx(self, idx):
        raise NotImplementedError()

    def translate(self, idx_or_word):
        if type(idx_or_word) is int:
            try:
                return self._from_idx(idx_or_word)
            except Exception:
                return UNKNOWN
        else:
            try:
                return self._to_idx(idx_or_word)
            except Exception:
                return -1


class CharVocab(VocabularyBase):
    def __init__(self, dim=10):
        self._dim = dim
        self._idx_to_chr = [PAD] + [chr(i) for i in range(128)]
        self._char_embed = {c: i for i, c in enumerate(self._idx_to_chr)}  # ASCII characters to idx vocab

    @property
    def embeddings_mx(self):
        return np.zeros((128, self._dim))

    @property
    def embedding_dim(self):
        return self._dim

    def __len__(self):
        return len(self._char_embed)

    def _to_idx(self, char):
        return self._char_embed[char]

    def _from_idx(self, idx):
        return self._idx_to_chr[idx]


class CostumeVocabulary(VocabularyBase):
    def __init__(self, list_tags, dim=20):
        self._dim = dim
        self._idx_to_tag = list(set(list_tags))
        self._tag_to_idx = {tag: i for i, tag in enumerate(self._idx_to_tag)}  # ASCII characters to idx vocab
        self._len = len(self._tag_to_idx)

    @property
    def embeddings_mx(self):
        return np.zeros((self._len, self._dim))

    @property
    def embedding_dim(self):
        return self._dim

    def __len__(self):
        return self._len

    def _to_idx(self, tag):
        return self._tag_to_idx[tag]

    def _from_idx(self, idx):
        return self._tag_to_idx[idx]
