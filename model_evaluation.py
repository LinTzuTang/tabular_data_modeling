from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve, auc
import numpy as np


def evalution_metrics(test_label, labels_score):
    """evaluate model performance

    Args:
        test_label (list): real test labels
        labels_score (list): prediction scores

    Returns:
        dict: metrics (accuracy, precision, sensitivity, specificity, f1, mcc)
    """

    # thres = 0.3
    # label_score = label_score - thres
    # label_score = np.ceil(label_score)

    accuracy = accuracy_score(test_label, labels_score.round())
    confusion = confusion_matrix(test_label, labels_score.round())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP / float(TP + FP)
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    f1 = f1_score(test_label, labels_score.round(),average='micro')
    mcc = matthews_corrcoef(test_label, labels_score.round())
    # precision TP / (TP + FP)
    # recall: TP / (TP + FN)
    # specificity : TN / (TN + FP)
    # f1: 2 TP / (2 TP + FP + FN)
    metrics = np.round([accuracy, precision, sensitivity, specificity, f1, mcc], 4)
    columns = ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'mcc']
    metrics_dict = dict(zip(columns, metrics))
    return (metrics_dict)


def show_histroy(df, train_acc='acc', validation_acc='val_acc', train_loss='loss', validation_loss='val_loss', path=''):
    """plot learning curve

    Args:
        df (df): learning_curve_df. from acc_loss_recorder function 
        train_acc (str, optional): Defaults to 'acc'.
        validation_acc (str, optional):  Defaults to 'val_acc'.
        train_loss (str, optional):  Defaults to 'loss'.
        validation_loss (str, optional):  Defaults to 'val_loss'.
        path (str, optional): where to save the learning curve plot. Defaults to ''.
    """
    fig1 = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[0, 1])
    #
    ax1.set_title('Train Accuracy', fontsize='14')
    ax2.set_title('Train Loss', fontfamily='serif', fontsize='18')
    ax1.set_xlabel('Epoch', fontfamily='serif', fontsize='13')
    ax1.set_ylabel('Acc', fontfamily='serif', fontsize='13')
    ax2.set_xlabel('Epoch', fontfamily='serif', fontsize='13')
    ax2.set_ylabel('Loss', fontfamily='serif', fontsize='13')
    ax1.plot(df['acc'], label='train', linewidth=2)
    ax1.plot(df['val_acc'], label='validation', linewidth=2)
    ax2.plot(df['loss'], label='train', linewidth=2)
    ax2.plot(df['val_loss'], label='validation', linewidth=2)
    ax1.legend(['train', 'validation'], loc='upper left')
    ax2.legend(['train', 'validation'], loc='upper left')
    fig1.savefig(path + 'history.png')
    plt.show()
