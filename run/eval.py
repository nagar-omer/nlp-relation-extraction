import sys
sys.path.insert(0, "..")
import os
import en_core_web_sm
from utils.data_preprocess.annotation_loader import AnnotationLoader
from utils.params import TRAIN_ANNOTATION_SRC, DEV_ANNOTATION_SRC
NLP = en_core_web_sm.load()


def _get_np_root(np_phrase):
    for w in NLP(np_phrase):
        if w.dep_ == "ROOT":
            return w.text


# because annotations gold label and np chunks can be slightly different
# e.g gold value is: "Air Line Pilots Association" and the np chunk value is "The Air Line Pilots Association"
# we tell if there are referring to the same object by looking at the ROOT word at the NP-chunk/Gold-val
def _get_label(labels, sent_id, per, org):
    label = 0
    org_root = _get_np_root(org)
    # loop over list of [ .. (per, org, org_root) .. ]
    for sent_per, sent_org, sent_org_root in labels.get(sent_id, []):
        # check similarity between PERSON and ORGANIZATION-ROOT
        if _get_np_root(per) == _get_np_root(sent_per) and org_root == sent_org_root:
            label = 1
            break
    return label


def _measure_success(positive, annotation):
    #  precision =  TP / (TP + FP)
    #  recall =     TP / (TP + FN)
    #  F1 =        2TP / (2TP + FP + FN)
    TP, FP, FN = (1e-6, 1e-6, 1e-6)
    for sent_id, per_org_list in positive.items():
        for (sent_per, sent_org, sent_org_root) in per_org_list:
            if _get_label(annotation, sent_id, sent_per, sent_org) == 1:
                TP += 1
            else:
                FP += 1

    for sent_id, per_org_list in annotation.items():
        for (sent_per, sent_org, sent_org_root) in per_org_list:
            if _get_label(positive, sent_id, sent_per, sent_org) == 0:
                FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)
    return precision, recall, F1


if __name__ == "__main__":
    args = sys.argv
    args = [".\\train_new_re_model.py", "..\data\RE\DEV.annotations", "pred.annotation"]

    truth = args[1]
    pred = args[2]
    truth = AnnotationLoader(truth).labels
    pred = AnnotationLoader(pred).labels
    precision, recall, F1 = _measure_success(pred, truth)
    print("\n\n---------------------------------------------------------------------------" +
          "\nprecision=" + str(precision) +
          "\nrecall=" + str(recall) +
          "\nF1=" + str(F1) +
          "\n---------------------------------------------------------------------------\n\n")
    e = 0