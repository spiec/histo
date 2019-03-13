#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

import itertools
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (auc, average_precision_score, classification_report, confusion_matrix,
                             precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve)
import keras.models

import config

# ----------------------------------------------------------------------
def training_summary(history, model, train_generator, eval_generator):
    """Report displayed after training.
    """
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12))
    fig.suptitle("Training Summary")
    axes = np.ravel(axes)

    keys = history.history.keys()
    print(keys)

    axes[0].plot(history.history["loss"], label="training loss", c="blue")
    axes[0].plot(history.history["val_loss"], label="validation loss", c="green")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="best")

    axes[1].plot(history.history["acc"], label="training acc", c="blue")
    axes[1].plot(history.history["val_acc"], label="validation acc", c="green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend(loc="best")

    eval_model(model, train_generator, eval_generator, axes[2:])
    plt.show()

# ----------------------------------------------------------------------
def eval_model(model, train_generator, eval_generator, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Model Evalution")
        axes = np.ravel(axes)

    y_true_train, y_pred_train = _run_model(model, train_generator)
    y_true_eval, y_pred_eval = _run_model(model, eval_generator)

    plot_roc(y_true_train, y_pred_train, ax=axes[0], label="Train ROC", c="red")
    plot_roc(y_true_eval, y_pred_eval, ax=axes[0], label="Eval ROC", c="darkorange")
    
    plot_precision_recall(y_true_train, y_pred_train, ax=axes[1])
    plot_precision_recall(y_true_eval, y_pred_eval, ax=axes[1])

    # confusion matrix (depends on the decision threshold)
    threshold = _optimal_threshold(y_true_train, y_pred_train, metric="auc")
    y_ = (y_pred_train > threshold).astype(int)

    cnf_matrix = confusion_matrix(y_true_train, y_)
    plot_confusion_matrix(cnf_matrix, threshold, labels=["0", "1"], ax=axes[2])
    print("Training set\n", classification_report(y_true_train, y_))

    threshold = _optimal_threshold(y_true_eval, y_pred_eval, metric="auc")
    y_ = (y_pred_eval > threshold).astype(int)

    cnf_matrix = confusion_matrix(y_true_eval, y_)
    plot_confusion_matrix(cnf_matrix, threshold, labels=["0", "1"], ax=axes[3])
    print("Evaluation set\n", classification_report(y_true_eval, y_))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Classification Examples") 

    #delta = y_true_eval - y_pred_eval
    plt.show()

# ----------------------------------------------------------------------
def _run_model(model, data_generator):
    y_true = []
    y_pred = []

    for X_batch, y_batch in tqdm(data_generator, desc="Run on generator"):
        y_ = model.predict_on_batch(X_batch)
        y_true += list(y_batch[:, 0].flatten())
        y_pred += list(y_[:, 0].flatten())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    return y_true, y_pred

# ----------------------------------------------------------------------
def _optimal_threshold(y_true, y_pred, metric="auc"):
    metric_map = {"auc": roc_auc_score,
                  "precision": precision_score,
                  "recall": recall_score
                  }

    best_score = -10e+7
    best_t = 0.

    # do it better
    for t in np.linspace(0, 1.0, 100):
        y_ = (y_pred > t).astype(int)
        score = metric_map[metric](y_true, y_)
        if score > best_score:
            best_score = score
            best_t = t

    return best_t

# ----------------------------------------------------------------------
def plot_confusion_matrix(cnf_matrix, threshold, labels, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues, ax=None):
    """
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if ax is None:
        ax = plt.gca()
    
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    ax.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title + " (threshold {:.2f})".format(threshold))
    #ax.colorbar()
    tick_marks = np.arange(len(labels))

    ax.set_xticklabels(labels)      #, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_xticks(tick_marks)      #, rotation=45)
    ax.set_yticks(tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        ax.text(j, i, format(cnf_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if cnf_matrix[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()

# ----------------------------------------------------------------------
def plot_roc(y_true, y_pred, ax=None, label="", *args, **kwargs):
    """
    """
    if ax is None:
        ax = plt.gca()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)        #, pos_label=1)

    ax.plot(fpr, tpr, label="{0} (AUC={1:.6f})".format(label, auc(fpr, tpr)), *args, **kwargs)
    ax.plot([0, 1], [0, 1], c="blue", linestyle="-.", alpha=0.2)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    ax.legend(loc="lower right")

# ----------------------------------------------------------------------
def plot_precision_recall(y_true, y_pred, label="", return_preds=False, ax=None):
    """ 
    """ 
    if ax is None:
        ax = plt.gca()
    
    df_true = pd.DataFrame(y_true)
    df_pred = pd.DataFrame(y_pred)

    preds = pd.concat([df_true, df_pred], axis=1)
    preds.columns = ["true_label", "pred_label"]

    preds.fillna(0.0, inplace=True)             # ? TODO
    #print(preds.sample(30))

    precision, recall, thresholds = precision_recall_curve(preds["true_label"], preds["pred_label"])
    avg_precision = average_precision_score(preds["true_label"], preds["pred_label"])

    ax.step(recall, precision, color='k', alpha=0.7, where='post',
            label="{0} avg precision: {1:.2f}".format(label, avg_precision))
    ax.fill_between(recall, precision, step='post', alpha=0.2, color='k')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])

    ax.set_title("Precision-recall curve (Avg precision: {0:.2f})".format(avg_precision))

# ----------------------------------------------------------------------
def inference(model, data_generator):
    data = []
    for X_batch, ids in tqdm(data_generator, desc="Predict on test"):
        # TODO
        # - test-time-augmentation
        # - committee of models

        y_ = list(model.predict_on_batch(X_batch)[:, 0].flatten())
        data += (zip(ids, y_))

    output_file = config.submission
    df = pd.DataFrame(data, columns=["id", "label"])
    df.to_csv(output_file, index=False)
    
# ----------------------------------------------------------------------
#def load_model(filename):
#    print("Loading model {}".format(filename))
#    return keras.models.load_model(filename)   
