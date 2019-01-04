import sys
sys.path.insert(0, "..")
from bokeh.plotting import figure, save
from bokeh.resources import Resources
import os
from utils.deep_utils.re_model_activator import REActivator
from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, TRAIN_ANNOTATION_SRC, ChrLevelLSTMParams, EncoderDecoderParams \
    , TopLayerParams, REFullModelParams, DEV_SRC, DEV_ANNOTATION_SRC, REActivatorParams
from utils.data_preprocess.corpus_loader import CorpusLoader
from utils.data_preprocess.glove_loader import GloVeLoader
from utils.data_preprocess.annotation_loader import AnnotationLoader
from utils.data_preprocess.relation_extraction_dataset import RelationExtractionDataset
from utils.deep_models.RE_full_model import TopLayerModel, REModel
import pickle


def get_x_y_axis(curve):
    x_axis = []
    y_axis = []
    for x, y in curve:
        x_axis.append(x)
        y_axis.append(y)
    return x_axis, y_axis


def _plot_loss_and_acc(train_line, dev_line, header, file_name, color_train='red', color_dev='orange', legend=""):
    if "fig" not in os.listdir(os.path.join("..")):
        os.mkdir(os.path.join("..", "fig"))
    p = figure(plot_width=600, plot_height=250, title="SNLI - Train/Dev " + header,
               x_axis_label="epochs", y_axis_label=header)

    x1, y1 = get_x_y_axis(train_line)
    x2, y2 = get_x_y_axis(dev_line)
    p.line(x1, y1, line_color=color_train, legend=legend + " Train")
    p.line(x2, y2, line_color=color_dev, legend=legend + " Dev")

    p.legend.background_fill_alpha = 0.5
    save(p, os.path.join("..", "fig", file_name + ".html"),
         title=header + ".html", resources=Resources(mode="inline"))


if __name__ == "__main__":
    args = sys.argv
    args = [".\train_new_re_model.py", "RE_best_loss_model", "..\data\RE\Corpus.TRAIN.txt",
            "..\data\RE\TRAIN.annotations", "..\data\RE\Corpus.DEV.txt",
            "..\data\RE\DEV.annotations", "..\data\GloVe\glove.6B.50d.txt"]
    model_name = args[1]
    train_corpus = args[2]
    train_annotation = args[3]
    dev_corpus = args[4]
    dev_annotation = args[5]
    pre_trained = args[6]

    # data
    # corpus_train = CorpusLoader(os.path.join("..", "..", TRAIN_SRC)).samples
    # corpus_dev = CorpusLoader(os.path.join("..", "..", DEV_SRC)).samples
    # labels_train = AnnotationLoader(os.path.join("..", "..", TRAIN_ANNOTATION_SRC)).labels
    # labels_dev = AnnotationLoader(os.path.join("..", "..", DEV_ANNOTATION_SRC)).labels
    # glove = GloVeLoader(os.path.join("..", "..", PRE_TRAINED_SRC))
    corpus_train = CorpusLoader(train_corpus).samples
    corpus_dev = CorpusLoader(dev_corpus).samples
    labels_train = AnnotationLoader(train_annotation).labels
    labels_dev = AnnotationLoader(dev_annotation).labels
    glove = GloVeLoader(pre_trained)
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

    # save results and model
    _plot_loss_and_acc(re_activator.loss_vec_train, re_activator.loss_vec_dev,
                       "Loss Train/Dev", model_name + "_loss_line", color_train='red', color_dev='orange',
                       legend="loss")
    _plot_loss_and_acc(re_activator.precision_vec_train, re_activator.precision_vec_dev,
                       "Precision Train/Dev", model_name + "_precision_line", color_train='green', color_dev='blue',
                       legend="precision")
    _plot_loss_and_acc(re_activator.recall_vec_train, re_activator.recall_vec_dev,
                       "Recall Train/Dev", model_name + "_recall_line", color_train='green', color_dev='blue',
                       legend="recall")
    _plot_loss_and_acc(re_activator.f1_vec_train, re_activator.f1_vec_dev,
                       "F1 Train/Dev", model_name + "_f1_line", color_train='green', color_dev='blue', legend="F1")
    pickle.dump(re_activator.best_model, open(os.path.join("..", "pkl", "trained_models", model_name + ".re_model"),
                                              "wb"))



