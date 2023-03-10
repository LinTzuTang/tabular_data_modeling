from sklearn.metrics import accuracy_score
from model_evaluation import evalution_metrics, show_histroy
import scipy.special
import numpy as np
import pandas as pd
import json
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import rtdl
import zero
from sam import SAM


def load_json(json_path):
    """load json file from given path"""
    with open(json_path) as data_file:
        loaded_json = json.load(data_file)
    return loaded_json


def split_X_y_pos_neg(X_splited, y_splited, sets="train"):
    """split X_splited and y_splited

    Args:
        X_splited (dict): loaded X_splited.json. get X_splited.json by the function train_val_test_split  in split_data_and_scaler.py
        y_splited (dict): loaded y_splited.json. get y_splited.json by the function train_val_test_split  in split_data_and_scaler.py
        sets (str, optional): train, val, test. Defaults to "train".

    Returns:
        4 lists: X_train_pos, X_train_neg, y_train_pos, y_train_neg
    """
    X_train_pos = []
    X_train_neg = []
    y_train_pos = []
    y_train_neg = []
    assert sets in ["train", "val", "test"]
    for x, y in zip(X_splited[sets], y_splited[sets]):
        if y == 0:
            X_train_neg.append(x)
            y_train_neg.append(y)
        else:
            X_train_pos.append(x)
            y_train_pos.append(y)
    print("pos: %s\n" % len(X_train_pos), "neg: %s\n" % len(X_train_neg))
    return X_train_pos, X_train_neg, y_train_pos, y_train_neg


def balance_pos_neg_data(X_train_pos, X_train_neg, y_train_pos, y_train_neg):
    """balance positive and negative data of imbalanced datasets

    Args:
        X_train_pos (list): X_pos (fewer than X_train_neg)
        X_train_neg (list): X_neg (more than X_train_pos)
        y_train_pos (list): y_pos
        y_train_neg (list): y_neg

    Returns:
        2 lists: X_train_combined, y_train_combined
    """
    # random split imbalanced data to balanced
    X_train_neg_s = random.sample(X_train_neg, len(X_train_pos))
    y_train_neg_s = [y_train_neg[0]] * len(X_train_neg_s)
    print("under sample to balanced: \n" + "pos: %s\n" % len(X_train_pos), "neg: %s\n" % len(X_train_neg_s))
    # combine pos and neg
    X_train_combined = X_train_neg_s + X_train_pos
    y_train_combined = y_train_neg_s + y_train_pos
    return X_train_combined, y_train_combined


def balance_splited_X_y_pos_neg(X_splited, y_splited, sets="train"):
    """balance X_splited and y_splited

    Args:
        X_splited (dict): loaded X_splited.json. get X_splited.json by the function train_val_test_split  in split_data_and_scaler.py
        y_splited (dict): loaded y_splited.json. get y_splited.json by the function train_val_test_split  in split_data_and_scaler.py
        sets (str, optional): train, val, test. Defaults to "train".

    Returns:
        2 dicts: X_splited, y_splited
    """
    assert sets in ["train", "val", "test"]
    X_train_pos, X_train_neg, y_train_pos, y_train_neg = split_X_y_pos_neg(
        X_splited, y_splited, sets=sets
    )
    X_train_combined, y_train_combined = balance_pos_neg_data(
        X_train_pos, X_train_neg, y_train_pos, y_train_neg
    )
    X_splited[sets] = X_train_combined
    y_splited[sets] = y_train_combined
    return X_splited, y_splited


def transform_splited_data_to_tensor(data_splited, device='cpu'):
    """transform data in X_splited or y_splited to tensor format

    Args:
        data_splited (dict): X_splited or y_splited
        device (str, optional): torch.device. Defaults to 'cpu'.

    Returns:
        dict: tensor format for model training
    """
    data_splited = {
        k: (torch.tensor(v, device=torch.device(device)))
        for k, v in data_splited.items()
    }
    return data_splited


class EarlyStopper:
    """early stopper class"""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """early stop by validation loss """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def set_optimizer(model):
    """set the optimizer"""
    lr = 0.001
    weight_decay = 0.0
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    return optimizer


def set_SAM_optimizer(model):
    optimizer = SAM(model.parameters(), torch.optim.AdamW)
    return optimizer


def run_normal_iter(model, apply_model, x_batch, y_batch, optimizer, cat_col=12, weight=None):
    loss_fn = set_loss_fn(task_type="binclass")
    loss = loss_fn(
        apply_model(
            model,
            x_batch[:, :cat_col].to(torch.float32),
            x_batch[:, cat_col:].to(torch.int64),
        ).squeeze(1),
        y_batch,
        weight=weight,
    )
    loss.backward()
    optimizer.step()
    return loss, optimizer


def run_sam_iter(model, apply_model, x_batch, y_batch, optimizer, cat_col=12, weight=None):
    loss_fn = set_loss_fn(task_type="binclass")
    # first forward-backward pass
    loss = loss_fn(
        apply_model(
            model,
            x_batch[:, :cat_col].to(torch.float32),
            x_batch[:, cat_col:].to(torch.int64),
        ).squeeze(1),
        y_batch,
        weight=weight,
    )  # use this loss for any training statistics
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # second forward-backward pass
    loss_fn(
        apply_model(
            model,
            x_batch[:, :cat_col].to(torch.float32),
            x_batch[:, cat_col:].to(torch.int64),
        ).squeeze(1),
        y_batch,
        weight=weight,
    ).backward()  # make sure to do a full forward pass
    optimizer.second_step(zero_grad=True)

    return loss, optimizer


ITER_FNS = {
    "normal": run_normal_iter,
    "sam": run_sam_iter
}


def set_reduce_lr(optimizer, mode="min", factor=0.1, patience=10, verbose=True):
    """Reduce learning rate when a metric has stopped improving"""
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        verbose=verbose
    )
    return scheduler


def set_loss_fn(task_type='binclass'):
    """set the loss function"""
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == 'binclass'
        else F.cross_entropy
        if task_type == 'multiclass'
        else F.mse_loss
    )
    return loss_fn


def set_train_loader(X_splited, batch_size=32, device="cpu", shuffle=True):
    """set the train_loader"""
    train_loader = zero.data.IndexLoader(
        len(X_splited["train"]), batch_size=batch_size, device=device, shuffle=True
    )
    return train_loader


def get_prediction_target(model, X_splited, y_splited, part, cat_col=12):
    """get prediction values and targets

    Args:
        model (model): model 
        X_splited (dict): loaded X_splited.json. get X_splited.json by the function train_val_test_split  in split_data_and_scaler.py
        y_splited (dict): loaded y_splited.json. get y_splited.json by the function train_val_test_split  in split_data_and_scaler.py
        part (str):train val test
        cat_col (int, num):Defaults to 12.

    Returns:
        2 np array: prediction values and targets
    """
    prediction = []
    for batch in zero.iter_batches(X_splited[part], 64):
        prediction.append(
            apply_model(model, batch[:, :cat_col].to(torch.float32), batch[:, cat_col:].to(torch.int64))
        )
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y_splited[part]
    # if task_type == "binclass":
    prediction = np.round(scipy.special.expit(prediction))
    # elif task_type == "multiclass":
    #    prediction = prediction.argmax(1)
    # else:
    #    assert task_type == "regression"
    #    score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
    return prediction, target


def acc_loss(model, X_splited, y_splited, loss_fn, part, cat_col, class_weights):
    prediction, target = get_prediction_target(model, X_splited, y_splited, part, cat_col=cat_col)
    acc = accuracy_score(target, prediction)
    loss = 0
    size = 0
    for batch, batch_y in zero.iter_batches((X_splited[part], y_splited[part]), 64):
        weight = torch.where(batch_y == 0, class_weights[0], class_weights[1])
        loss += loss_fn(
            apply_model(model,
                        batch[:, :cat_col].to(torch.float32), batch[:, cat_col:].to(torch.int64)
                        ).squeeze(1),
            batch_y,
            weight=weight,
        ).item() * len(batch)
        size += len(batch)
    loss = loss / size
    return acc, loss


@torch.no_grad()
def evaluate(model, X_splited, y_splited, part, cat_col):
    model.eval()
    prediction, target = get_prediction_target(model, X_splited, y_splited, part, cat_col=cat_col)
    metrics = evalution_metrics(target, prediction)
    return metrics


@torch.no_grad()
def acc_loss_recorder(model, X_splited, y_splited, loss_fn, epoch, cat_col, class_weights):
    model.eval()
    acc, loss = acc_loss(model, X_splited, y_splited, loss_fn, "train", cat_col, class_weights)
    val_acc, val_loss = acc_loss(model, X_splited, y_splited, loss_fn, "val", cat_col, class_weights)
    df = pd.DataFrame(
        {"acc": acc, "val_acc": val_acc, "loss": loss, "val_loss": val_loss},
        index=[epoch],
    )
    return df


def apply_model(model, x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f"Looks like you are using a custom model: {type(model)}."
            " Then you have to implement this branch first."
        )
