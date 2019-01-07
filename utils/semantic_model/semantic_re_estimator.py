import en_core_web_sm
from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
PER_FTR = "ATT_PER"
ORG_FTR = "ATT_PER"
import numpy as np


"""
the following class uses data that is generated with the RelationExtractionDataset class
the data consist of triplets, (person, organization, sentence) and is labeled 1 if the data represent Work_for relation
and 0 otherwise
the goal of this class is to use semantics to predict whether a new sample represents Work_For relation or not without
using any deep learning

operation:
    PART 1 -COUNT
    - the class will follow the parents from ORG and PER up to first intersection
    - during that process the class will count the semantic relation
        - two types of counters will be kept, counter for positive samples and counter for total att count
    - we will also count the tree height up to the intersection

    PART 2 -CALC PROBABILITIES
    - create probabilities for each attribute att = att_count / total
    - use gmm to create probability for the tree height
    
    PART 3 -PREDICT
    - use probabilities to predict whether a new sample represents Work_For relation or not 
"""


class SemanticEstimator:
    def __init__(self, dataset: RelationExtractionDataset):
        self._nlp = en_core_web_sm.load()
        self._p_positive, self._p_ftr_given_work, self._p_ftr, self._mue, self._sigma = \
            self._count_probabilities(dataset)

    def load_nlp(self):
        self._nlp = en_core_web_sm.load()

    def suspend_nlp(self):
        self._nlp = None

    def _prob_by_tree_height(self, x):
        e = np.exp(-((x - self._mue)**2) / (2 * (self._sigma ** 2)))
        v = 1 / (6.283185 * (self._sigma ** 2)) ** 0.5
        return v * e

    def _get_np_root(self, np_phrase, idx):
        idx = idx[0]
        for w in self._nlp(np_phrase.text):
            if w.dep_ == "ROOT":
                return w.text, w.i + idx

    def _get_ftr_list(self, sent, per, org, per_idx, org_idx):
        # get root idx with respect to the whole sentence
        per_root, per_idx = self._get_np_root(per, per_idx)
        org_root, org_idx = self._get_np_root(org, org_idx)
        sent = [w for w in self._nlp(" ".join(sent))]

        # get path in the semantic tree from PER/ORG up to the head node
        per_tree, org_tree = [], []
        curr_per_idx, curr_org_idx = per_idx, org_idx
        root = False
        while not root:
            per_tree.append(curr_per_idx)
            root = True if sent[curr_per_idx].dep_ == "ROOT" else False
            curr_per_idx = sent[curr_per_idx].head.i
        root = False
        while not root:
            org_tree.append(curr_org_idx)
            root = True if sent[curr_org_idx].dep_ == "ROOT" else False
            curr_org_idx = sent[curr_org_idx].head.i

        ftr_list = []
        if not list(set(org_tree) & set(per_tree)):
            org_tree.append(per_tree[-1])
            ftr_list.append((PER_FTR, ORG_FTR))
            return ftr_list
        # add ftr form the PER/ORG tree parents until intersection

        while per_idx not in org_tree:
            ftr_list.append((PER_FTR, sent[per_idx].dep_))
            per_idx = sent[per_idx].head.i
        while org_idx not in per_tree:
            ftr_list.append((PER_FTR, sent[org_idx].dep_))
            org_idx = sent[org_idx].head.i

        return ftr_list

    def _first_common_node(self, indices0, indices1):
        for i in indices0:
            if i in indices1:
                return i

    def _count_probabilities(self, dataset):
        # our final goal is to calculate
        #                                               P( ftr_1, ... , ftr_n | label=Work_For ) * P( label=Work_For )
        # P( label=Work_For | ftr_1, ... , ftr_n ) =    --------------------------------------------------------------
        #                                                                   P( ftr_1, ... , ftr_n)
        #
        # P( label=Work_For ) =                         #( label=Work_For ) / #( label=# )
        # P( ftr_1, ... , ftr_n | label=Work_For ) =    multiply (#(ftr_i & label=Work_For) / #(label=Work_For))
        # P( ftr_1, ... , ftr_n) =                      multiply (#(ftr_i) / #(total_ftr_count))

        positive_att = {}               # #(ftr_i & label=Work_For)
        att_count = {}                  # #(ftr_i)
        total_att_count = 0             # #(total_ftr_count)
        total_samples = 0               # #( label=# )
        positive_count = 0              # #( label=Work_For )
        positive_tree_lengths = []

        for sent_id, (per, per_idx), (org, org_idx), (words, tree_parent), label in dataset.semantic_samples:
            ftr_list = self._get_ftr_list(words, per, org, per_idx, org_idx)
            # count
            total_samples += 1                                          # #( label=# )
            for ftr in ftr_list:
                total_att_count += 1                                    # #(total_ftr_count)
                att_count[ftr] = att_count.get(ftr, 0) + 1              # #(ftr_i)

            if label == 1:
                positive_tree_lengths.append(len(ftr_list))
                positive_count += 1                                     # #( label=Work_For )
                for ftr in ftr_list:
                    positive_att[ftr] = positive_att.get(ftr, 0) + 1    # #(ftr_i & label=Work_For)

        p_positive = positive_count / total_samples                                 # P( label=Work_For )
        p_ftr_given_work = {ftr: positive_att.get(ftr, 1e-3) / positive_count for ftr in att_count}  # P(ftr_i|Work_For)
        p_ftr = {ftr: count / total_samples for ftr, count in att_count.items()}    # P( ftr_i )
        return p_positive, p_ftr_given_work, p_ftr, np.mean(positive_tree_lengths), np.std(positive_tree_lengths)

    def _pred_by_ftr_list(self, ftr_list):
        #                                               P( ftr_1, ... , ftr_n | label=Work_For ) * P( label=Work_For )
        # P( label=Work_For | ftr_1, ... , ftr_n ) =    --------------------------------------------------------------
        #                                                                   P( ftr_1, ... , ftr_n)
        # pred = self._prob_by_tree_height(len(ftr_list))    # add probability by tree height
        p_ftr_given_work_for = 1e-3
        p_ftr = 1e-3
        for ftr in ftr_list:
            if self._p_ftr_given_work.get(ftr, 0) > p_ftr_given_work_for:
                p_ftr_given_work_for = self._p_ftr_given_work[ftr]
                p_ftr = self._p_ftr[ftr]
        return [p_ftr_given_work_for * self._p_positive / p_ftr,
                self._prob_by_tree_height(len(ftr_list)) / self._prob_by_tree_height(self._mue)]

    def predict(self, ds):
        all_pred = []
        for sent_id, (per, per_idx), (org, org_idx), (words, tree_parent), label in ds.semantic_samples:
            ftr_list = self._get_ftr_list(words, per, org, per_idx, org_idx)
            pred = self._pred_by_ftr_list(ftr_list)
            all_pred.append([sent_id, " ".join(per.text.split()), " ".join(org.text.split()),
                             " ".join(words), pred])
        # normalize
        # m = max([i[4][0] for i in all_pred])
        # for i in range(len(all_pred)):
        #     all_pred[i][4][0] /= m
        return all_pred


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC, DEV_SRC, TRAIN_ANNOTATION_SRC, DEV_ANNOTATION_SRC, PRE_TRAINED_SRC
    from utils.data_preprocess.annotation_loader import AnnotationLoader
    from utils.data_preprocess.glove_loader import GloVeLoader
    from utils.data_preprocess.corpus_loader import CorpusLoader
    # data
    corpus_train = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    corpus_dev = CorpusLoader(os.path.join("..", "..", DEV_SRC)).samples
    labels_train = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    labels_dev = AnnotationLoader(os.path.join("..", "..", DEV_ANNOTATION_SRC)).labels
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))

    ds_train = RelationExtractionDataset(corpus_train, glove, labels=labels_train)
    ds_dev = RelationExtractionDataset(corpus_dev, glove, labels=labels_dev)

    # model
    estimator = SemanticEstimator(ds_train)
    print(estimator.predict(ds_dev))
    e = 0







