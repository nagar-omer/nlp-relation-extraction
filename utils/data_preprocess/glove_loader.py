import pickle
import numpy as np
import os
from utils.params import PAD, UNKNOWN
from utils.data_preprocess.vocabulary import VocabularyBase


class GloVeLoader(VocabularyBase):
    def __init__(self, src_file, dim=50):
        self._dim = dim
        self._base_dir = __file__.replace("/", os.sep)  # absolute path to base project dir
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..", "..")
        self._word_to_idx, self._idx_to_word, self._embed_mx = self._read_glove_file(src_file)

    @property
    def embeddings_mx(self):
        return self._embed_mx

    @property
    def embedding_dim(self):
        return self._dim

    def __len__(self):
        return len(self._idx_to_word)

    def _to_idx(self, word):
        return self._word_to_idx[word]

    def _from_idx(self, idx):
        return self._idx_to_word[idx]

    # read vocabulary + vectors from file (or from pkl if possible)
    def _read_glove_file(self, vocab_src):
        # load pickle if exists
        pkl_path = os.path.join(self._base_dir, "pkl", vocab_src.rsplit(os.sep, 1)[1].strip(".txt") + ".pkl")
        if os.path.exists(pkl_path):
            return pickle.load(open(pkl_path, "rb"))

        vocab_list = [PAD, UNKNOWN]  # fixed tags
        mx_list = [np.zeros(self._dim), np.zeros(self._dim)]  # init vectors to zero vectors
        src = open(vocab_src, "rt", encoding="utf-8")
        for row in src:
            word, vec = row.split(" ", 1)
            mx_list.append(np.fromstring(vec, sep=" "))  # append pre trained vector
            vocab_list.append(word)  # append word
        mx = np.vstack(mx_list)  # concat vectors

        word_to_idx = {word: i for i, word in enumerate(vocab_list)}
        # save as pickle
        pickle.dump((word_to_idx, vocab_list, mx), open(pkl_path, "wb"))
        return word_to_idx, vocab_list, mx


if __name__ == "__main__":
    import os
    from utils.params import PRE_TRAINED_SRC
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))
    e = 0