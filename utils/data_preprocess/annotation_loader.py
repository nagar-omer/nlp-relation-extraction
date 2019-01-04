import codecs
import en_core_web_sm
from utils.params import RELATION


class AnnotationLoader:
    def __init__(self, src_file):
        self._nlp = en_core_web_sm.load()
        self._labels = self._read_corpus_file(src_file)

    @staticmethod
    def _read_lines(f_name):
        for line in codecs.open(f_name, encoding="utf8"):
            sent_id, obj1, relation, obj2, sent = line.strip().split("\t")
            if relation != RELATION:
                continue
            yield sent_id, obj1, obj2

    def _get_np_root(self, np_phrase):
        for w in self._nlp(np_phrase):
            if w.dep_ == "ROOT":
                return w.text

    def _read_corpus_file(self, src_file):
        labels = {}
        for sent_id, per, org in AnnotationLoader._read_lines(src_file):
            labels[sent_id] = labels.get(sent_id, []) + [(per, org, self._get_np_root(org))]
        return labels

    @property
    def labels(self):
        return self._labels


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_ANNOTATION_SRC
    labels = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    e = 0
