import codecs


class CorpusLoader:
    def __init__(self, src_file):
        self._samples = self._read_corpus_file(src_file)

    @staticmethod
    def _read_lines(f_name):
        src = open(f_name, encoding="utf8")
        for line in src:
            sent_id, sent = line.strip().split("\t")
            sent = sent.replace("-LRB-", "(").replace("-RRB-", ")")
            yield sent_id, sent

    @staticmethod
    def _read_corpus_file(src_file):
        samples = {}
        for i, (seq_id, seq) in enumerate(CorpusLoader._read_lines(src_file)):
            samples[seq_id] = seq
        print(i, "samples in corpus")
        return samples

    @property
    def samples(self):
        return self._samples


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC
    corpus = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    e = 0
