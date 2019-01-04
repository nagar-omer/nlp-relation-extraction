import en_core_web_sm
import torch
from torch.autograd import Variable
from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
from utils.deep_models.RE_full_model import REModel
from utils.deep_utils.imbalnced_data_sampler import ImbalancedDatasetSampler
from utils.logger.loggers import PrintLogger
from utils.params import REActivatorParams, DEV_ANNOTATION_SRC, DEV_SRC
from sys import stdout
from copy import deepcopy
from torch.utils.data import DataLoader


class REActivator:
    def __init__(self, model: REModel, params: REActivatorParams, train: RelationExtractionDataset=None,
                 dev: RelationExtractionDataset=None, train_annotation=None, dev_annotation=None):
        self._dev_annotation = dev_annotation
        self._train_annotation = train_annotation
        self._nlp = en_core_web_sm.load()
        self._model = model
        self._epochs = params.EPOCHS
        self._validation_rate = params.VALIDATION_RATE
        self._batch_size = params.BATCH_SIZE
        self._gpu = params.GPU
        self._loss_func = params.LOSS
        if self._gpu:
            self._model.cuda()
        self._load_data(train, dev)
        self._init_loss_and_acc_vec()
        self._best_model = model

    def _get_np_root(self, np_phrase):
        for w in self._nlp(np_phrase):
            if w.dep_ == "ROOT":
                return w.text

    @property
    def best_model(self):
        return self._best_model

    @property
    def model(self):
        return self._model

    # load dataset
    def _load_data(self, train_dataset, dev_dataset):
        self._dev_loader = None
        self._train_loader = None
        # set train loader
        if train_dataset is not None:
            self._train_loader = DataLoader(
                train_dataset,
                batch_size=self._batch_size,
                collate_fn=train_dataset.collate_fn,
                sampler=ImbalancedDatasetSampler(train_dataset),
            )
            self._train_validation_loader = DataLoader(
                train_dataset,
                batch_size=self._batch_size,
                collate_fn=train_dataset.collate_fn,
                shuffle=True
            )
        # set validation loader
        if dev_dataset is not None:
            self._dev_loader = DataLoader(
                dev_dataset,
                batch_size=self._batch_size,
                collate_fn=train_dataset.collate_fn,
                shuffle=True
            )

    def _init_loss_and_acc_vec(self):
        self._best_model_f1 = 0
        self._best_model_loss = 1
        self._best_model_precision = 0
        self._best_model_recall = 0
        self.loss_vec_dev = []
        self.precision_vec_dev = []
        self.recall_vec_dev = []
        self.f1_vec_dev = []
        self.loss_vec_train = []
        self.precision_vec_train = []
        self.recall_vec_train = []
        self.f1_vec_train = []

    def _validate_train_and_dev(self, epoch_num):
        with torch.no_grad():
            # validate Train
            if self._train_annotation is not None:
                loss, precision, recall, F1 = self._validate(self._train_validation_loader, self._train_annotation,
                                                             job="Train")
                self.loss_vec_train.append((epoch_num, loss))
                self.precision_vec_train.append((epoch_num, precision))
                self.recall_vec_train.append((epoch_num, recall))
                self.f1_vec_train.append((epoch_num, F1))
            # validate Dev
            if self._dev_loader is not None:
                loss, precision, recall, F1 = self._validate(self._dev_loader, self._dev_annotation, job="Dev")
                self.loss_vec_dev.append((epoch_num, loss))
                self.precision_vec_dev.append((epoch_num, precision))
                self.recall_vec_dev.append((epoch_num, recall))
                self.f1_vec_dev.append((epoch_num, F1))
                # save best model
                if loss < self._best_model_loss:
                    self._best_model = deepcopy(self._model)
                    self._best_model_f1 = F1
                    self._best_model_loss = loss
                    self._best_model_precision = precision
                    self._best_model_recall = recall

    def _measure_success(self, positive, negative, annotation):
        #  precision =  TP / (TP + FP)
        #  recall =     TP / (TP + FN)
        #  F1 =        2TP / (2TP + FP + FN)
        TP, FP, TN, FN = (1e-6, 1e-6, 1e-6, 1e-6)
        for sent_id, (pred_per, pred_org, pred_org_root) in positive.items():
            true_positive = False
            for real_per, real_org, real_org_root in annotation.get(sent_id, []):
                # check similarity between PERSON and ORGANIZATION-ROOT
                if pred_per == real_per and pred_org_root == real_org_root:
                    true_positive = True
                    break
            if true_positive:
                TP += 1
            else:
                FP += 1

        for sent_id, (pred_per, pred_org, pred_org_root) in negative.items():
            false_negative = False
            for real_per, real_org, real_org_root in annotation.get(sent_id, []):
                # check similarity between PERSON and ORGANIZATION-ROOT
                if pred_per == real_per and pred_org_root == real_org_root:
                    false_negative = True
                    break
            if false_negative:
                FN += 1
            else:
                TN += 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * TP / (2 * TP + FP + FN)
        return precision, recall, F1

    # train a model, input is the enum of the model type
    def train(self):
        logger = PrintLogger("NN_train")
        if self._train_loader is None:
            logger.info("load train file to train model")
            return
        logger.info("start_train")
        self._init_loss_and_acc_vec()

        for epoch_num in range(self._epochs):
            logger.info("epoch:" + str(epoch_num))
            # set model to train mode
            self._model.train()
            # calc number of iteration in current epoch
            # len_data = len(self._train_loader)
            len_data = 20
            for batch_index, (sent_id, per, org, sent, label) in enumerate(self._train_loader):
                stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len_data) + "%")
                stdout.flush()

                self._model.zero_grad()                           # zero gradients
                output = self._model(per, org, sent, grad=False)  # calc output of current model on the current batch
                label = Variable(label).cuda() if self._gpu else Variable(label)
                loss = self._loss_func(output.squeeze(dim=1), label.float())     # calculate loss
                loss.backward()                                   # back propagation
                self._model.optimizer.step()                      # update weights

                if self._validation_rate and batch_index % self._validation_rate == 0:
                    logger.info("\nvalidating dev...    epoch:" + "\t" + str(epoch_num + 1) + "/" + str(self._epochs))
                    self._validate_train_and_dev(epoch_num + (batch_index / len_data))
                    self._model.train()
        logger.info("\n\n---------------------------------------------------------------------------" +
                    "\nFinalModel:" +
                    "\nloss=" + str(self._best_model_loss) +
                    "\nprecision=" + str(self._best_model_precision) +
                    "\nrecall=" + str(self._best_model_recall) +
                    "\nF1=" + str(self._best_model_f1) +
                    "\n---------------------------------------------------------------------------\n\n")

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, annotation, job=""):
        logger = PrintLogger(job + "_NN_validation")
        loss_count = 0
        positive = {}
        negative = {}
        self._model.eval()
        len_data = len(data_loader)
        for batch_index, (sent_id, per, org, sent, label) in enumerate(data_loader):
            stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len_data) + "%")
            stdout.flush()

            output = self._model(per, org, sent)
            # calculate total loss
            loss_count += self._loss_func(output.squeeze(dim=1), label.cuda().float() if self._gpu else label.float())

            # statistical measures
            per_words, _ = per
            org_words, _ = org
            sent_words, _, _, _, _, _ = sent
            for i, pred in enumerate(torch.round(output)):
                if int(pred.item()) == 1:
                    positive[sent_id[i]] = (" ".join(per_words[i]), " ".join(org_words[i]),
                                            self._get_np_root(" ".join(org_words[i])))
                else:
                    negative[sent_id[i]] = (" ".join(per_words[i]), " ".join(org_words[i]),
                                            self._get_np_root(" ".join(org_words[i])))

        loss = float(loss_count / len(data_loader))
        if self._dev_annotation is not None:
            precision, recall, F1 = self._measure_success(positive, negative, annotation)
            logger.info("loss=" + str(loss) + " - precision=" + str(precision) + " - recall=" + str(recall)
                        + " - F1=" + str(F1))
            return loss, precision, recall, F1
        # logger.info("loss=" + str(loss) + "  ---  accuracy=" + str(accuracy) + "  ---  recall=" + str(recall))
        return loss

    def predict(self, dataset: RelationExtractionDataset):
        self.model.eval()
        test_data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False
        )
        with torch.no_grad():
            all_pred = []
            self._model.eval()
            len_data = len(test_data_loader)
            for batch_index, (sent_id, per, org, sent, label) in enumerate(test_data_loader):
                stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len_data) + "%")
                stdout.flush()
                per_words, _ = per
                org_words, _ = org
                sent_words, _, _, _, _, _ = sent

                output = self._model(per, org, sent)
                # calculate accuracy and loss
                for i, pred in enumerate(torch.round(output)):
                    all_pred.append([sent_id[i], " ".join(per_words[i]), " ".join(org_words[i]), " ".join(sent_words[i])
                                    , int(pred.item())])
        return all_pred


if __name__ == "__main__":
    import os
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, TRAIN_ANNOTATION_SRC, ChrLevelLSTMParams, EncoderDecoderParams\
        , TopLayerParams, REFullModelParams
    from utils.data_preprocess.corpus_loader import CorpusLoader
    from utils.data_preprocess.glove_loader import GloVeLoader
    from utils.data_preprocess.annotation_loader import AnnotationLoader
    from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
    from utils.deep_models.RE_full_model import TopLayerModel

    # data
    corpus_train = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    corpus_dev = CorpusLoader(os.path.join("..", "..", DEV_SRC)).samples
    labels_train = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    labels_dev = AnnotationLoader(os.path.join("..", "..", DEV_ANNOTATION_SRC)).labels
    glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))
    ds_train = RelationExtractionDataset(corpus_train, glove, labels=labels_train)
    ds_dev = RelationExtractionDataset(corpus_dev, glove, labels=labels_dev)

    # params
    chr_model_params = ChrLevelLSTMParams()
    att_params = EncoderDecoderParams(len(ds_train.word_vocab), len(ds_train.pos_vocab), len(ds_train.ner_vocab),
                                      chr_model_params.OUT_DIM, pre_trained=ds_train.word_vocab.embeddings_mx)
    top_mlp_params = TopLayerParams(att_params.OUT_DIM)
    re_params = REFullModelParams(chr_model_params, att_params, top_mlp_params)
    activator_param = REActivatorParams()

    # models
    top_mlp = TopLayerModel(top_mlp_params)
    re_model = REModel(re_params)
    re_activator = REActivator(re_model, activator_param, ds_train, ds_dev, train_annotation=labels_train,
                               dev_annotation=labels_dev)
    re_activator.train()
    re_activator.predict(ds_dev)

