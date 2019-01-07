import sys
sys.path.insert(0, "..")
import os
import pickle
import sys
from utils.deep_utils.re_model_activator import REActivator
from utils.params import REActivatorParams
from utils.data_preprocess.corpus_loader import CorpusLoader
from utils.data_preprocess.annotation_loader import AnnotationLoader
from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
import numpy as np


def _fial_pred(deep_pred, semantic_pred, to_file=None):
    final_pred_list = []
    for (sent_id, per, org, sent, d_pred), (sent_id, per, org, sent, s_pred) in zip(deep_pred, semantic_pred):
        score = 0.6*d_pred + 0.05*s_pred[0] + 0.35*s_pred[1]
        if score > 0.5:
            final_pred_list.append((sent_id, per, org, sent))
    if to_file is None:
        return final_pred_list
    out = open(to_file, "wt")
    for sent_id, per, org, sent in final_pred_list:
        out.write(sent_id + "\t" + per + "\t" + "Work_For" + "\t" + org + "\t(" + sent + ")\n")


if __name__ == "__main__":
    args = sys.argv
    # args = [".\\train_new_re_model.py", "RE_best_loss_model", "..\data\RE\Corpus.DEV.txt", "pred.annotation"]

    model_name = args[1]
    test_corpus = args[2]
    out_name = args[3]

    re_model, glove, semantic_model = \
        pickle.load(open(os.path.join("..", "pkl", "trained_models", model_name + ".re_model"), "rb"))
    semantic_model.load_nlp()

    # data
    corpus_test = CorpusLoader(test_corpus).samples
    ds_test = RelationExtractionDataset(corpus_test, glove)
    re_activator = REActivator(re_model, REActivatorParams())

    deep_pred = re_activator.predict(ds_test)
    semantic_pred = semantic_model.predict(ds_test)
    _fial_pred(deep_pred, semantic_pred, to_file=out_name)
