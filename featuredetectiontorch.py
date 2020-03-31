#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import pathlib
import pickle
import time
from tqdm import tqdm
import argparse
import logging.handlers

import torch
import torchvision
from torch import nn, optim
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


DATASET_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/'
TRAIN_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_train_split/whole_resize'
TEST_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_eval_img_resize/'
BATCH_SIZE = 32
LOG_DIR = "logs"
FORMAT = "[%(levelname)s] %(message)s"
means = [0.5, 0.5, 0.5]
stds = [0.5, 0.5, 0.5]


def configure_logger(args):
    logs_path = os.path.join(args.model_name, LOG_DIR)
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join(logs_path, 'stdout.log'))
    file_handler.setFormatter(logging.Formatter(FORMAT))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def classifications(actuals, preds):
    tp = torch.sum((actuals == 1) & (preds == 1))
    tn = torch.sum((actuals == 0) & (preds == 0))
    fp = torch.sum((actuals == 0) & (preds == 1))
    fn = torch.sum((actuals == 1) & (preds == 0))

    return tp, tn, fp, fn


def precision(tp, tn, fp, fn):
    deno = tp + fp
    if deno.item() == 0:
        return 0.
    precision = tp.type(torch.float) / (tp + fp)
    return precision.item()


def recall(tp, tn, fp, fn):
    deno = tp + fn
    if deno.item() == 0:
        return 0.
    recall = tp.type(torch.float) / (tp + fn)

    return recall.item()


def f1score(actuals, preds):
    classi = classifications(actuals, preds)
    prec = precision(*classi)
    rec = recall(*classi)
    deno = prec + rec
    if deno == 0:
        return 0.
    f1score = 2 * prec * rec / (prec + rec)
    return f1score


def calculate_f1(prec, rec):
    deno = prec + rec
    if deno == 0:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return f1


def classifications_at_threshold(actuals, outputs, threshold):
    # The clone is very imp else it modifies the original tensor
    outputs = outputs.clone()
    preds = outputs.apply_(lambda x: 1 if x >= threshold else 0)
    tp = torch.sum((actuals == 1) & (preds == 1))
    tn = torch.sum((actuals == 0) & (preds == 0))
    fp = torch.sum((actuals == 0) & (preds == 1))
    fn = torch.sum((actuals == 1) & (preds == 0))

    return tp, tn, fp, fn


def calc_pres_recall(actuals, outputs):
    pres_list = []
    rec_list = []
    thresholds = [x / 100 for x in range(0, 101)]
    for threshold in thresholds:
        results = classifications_at_threshold(actuals, outputs, threshold)
        pres_list.append(precision(*results))
        rec_list.append(recall(*results))
    return pres_list, rec_list


def validate(args, model, val_gen, bce_loss):
    results = {}
    model.eval()
    all_preds = []
    all_labels = []
    tps, tns, fps, fns = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
    val_loss = 0.0
    for images, labels in val_gen:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        outputs = nn.Sigmoid()(outputs)
        loss = bce_loss(outputs, labels)
        val_loss += loss.cpu().item() * labels.size(0)
        preds = outputs.cpu().data.apply_(lambda x: 1 if x >= 0.5 else 0)
        all_labels.append(labels.detach().cpu())
        all_preds.append(outputs.detach().cpu())
        tp, tn, fp, fn = classifications(labels.cpu(), preds)
        tps += tp
        tns += tn
        fps += fp
        fns += fn
    val_auc = roc_auc_score(torch.cat(all_labels).flatten(), torch.cat(all_preds).flatten())
    val_loss = val_loss / (len(val_gen) * BATCH_SIZE)
    prec = precision(tps, tns, fps, fns)
    rec = recall(tps, tns, fps, fns)
    f1 = calculate_f1(prec, rec)
    results['val_loss'] = val_loss
    results['val_tps'] = tps.cpu().item()
    results['val_tns'] = tns.cpu().item()
    results['val_fps'] = fps.cpu().item()
    results['val_fns'] = fns.cpu().item()
    results['val_precision'] = prec
    results['val_recall'] = rec
    results['val_f1'] = f1
    results['val_auc'] = val_auc

    return results

def is_cuda():
    return torch.cuda.is_available()


def set_cuda_device():
    if is_cuda():
        cuda_visible = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_visible:
            torch.cuda.set_device('cuda:0')
        else:
            torch.cuda.set_device('cuda:1')
    else:
        torch.cuda.set_device('cpu')


def get_training_gen(training_df, all_features, test_split=0.25):
    train_df, val_df = train_test_split(training_df, test_size=test_split)
    training_transforms = Compose([ToTensor(), Normalize(means, stds)])
    train_ds = ImageFeaturesDataSet(train_df, all_features, TRAIN_IMAGE_PATH, transform=training_transforms)
    train_gen = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_ds = ImageFeaturesDataSet(val_df, all_features, TRAIN_IMAGE_PATH, transform=training_transforms)
    val_gen = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return train_gen, val_gen


def get_testing_gen(testing_df, all_features):

    testing_transforms = Compose([ToTensor(), Normalize(means, stds)])
    test_ds = ImageFeaturesDataSet(testing_df, all_features, TEST_IMAGE_PATH, transform=testing_transforms)
    test_gen = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return test_gen


def train(args, model, training_gen, val_gen):
    history = {'loss': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [],
               'val_loss': [], 'val_tp': [], 'val_tn': [], 'val_fp': [], 'val_fn': [], 'val_precision': [],
               'val_recall': [], 'val_f1': [], 'val_auc': []}

    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = torch.nn.BCELoss()
    if is_cuda():
        # torch.cuda.set_device('cuda:1')
        model.cuda()
    epochs = args.epochs

    for epoch in range(epochs):
        # print(f"epoch: {epoch+1} started...")
        start = time.time()
        model.train()
        train_loss = 0.0
        all_labels, all_preds = [], []
        tps, tns, fps, fns = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        for batch, (images, labels) in tqdm(enumerate(training_gen), desc=f'epoch: {epoch + 1} ',
                                            total=len(training_gen)):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            adam.zero_grad()
            outputs = model(images)
            outputs = nn.Sigmoid()(outputs)
            preds = outputs.cpu().data.apply_(lambda x: 1 if x >= 0.5 else 0)
            loss = bce_loss(outputs, labels)
            train_loss += loss.cpu().item() * labels.size(0)
            loss.backward()
            adam.step()
            tp, tn, fp, fn = classifications(labels.cpu(), preds)
            tps += tp
            tns += tn
            fps += fp
            fns += fn
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
        train_auc = roc_auc_score(torch.cat(all_labels).flatten(), torch.cat(all_preds).flatten())
        prec = precision(tps, tns, fps, fns)
        rec = recall(tps, tns, fps, fns)
        f1 = calculate_f1(prec, rec)
        with torch.no_grad():
            val_results = validate(args, model, val_gen, bce_loss)
        end = time.time()
        duration = (end - start) / 60
        train_loss = train_loss / (len(training_gen) * BATCH_SIZE)
        logger.info(f"epoch: {epoch + 1}, duration: {duration: 0.2f} mins")
        logger.info(
            f"loss: {train_loss: 0.4f}, precision: {prec: 0.4f}, recall: {rec: 0.4f}, f1: {f1: 0.4f}, auc: {train_auc: 0.4f}")
        logger.info(
            f"val_loss: {val_results['val_loss']: 0.4f}, val_precision: {val_results['val_precision']: 0.4f}, val_recall: {val_results['val_recall']: 0.4f}, val_f1: {val_results['val_f1']: 0.4f}, val_auc: {val_results['val_auc']: 0.4f}")
        history['loss'].append(train_loss)
        history['tp'].append(tps.cpu().item())
        history['tn'].append(tns.cpu().item())
        history['fp'].append(fps.cpu().item())
        history['fn'].append(fns.cpu().item())
        history['precision'].append(prec)
        history['recall'].append(rec)
        history['f1'].append(f1)
        history['auc'].append(train_auc)
        history['val_loss'].append(val_results['val_loss'])
        history['val_tp'].append(val_results['val_tps'])
        history['val_tn'].append(val_results['val_tns'])
        history['val_fp'].append(val_results['val_fps'])
        history['val_fn'].append(val_results['val_fns'])
        history['val_precision'].append(val_results['val_precision'])
        history['val_recall'].append(val_results['val_recall'])
        history['val_f1'].append(val_results['val_f1'])
        history['val_auc'].append(val_results['val_auc'])

    torch.save(model, f'{args.model_name}_model.pickle')
    logger.info(f"Model saved to {args.model_name}_model.pickle")
    return history


def evaluate_on_test_set(loaded_model, test_gen):
    all_preds = []
    all_labels = []
    loaded_model.eval()
    if torch.cuda.is_available():
        loaded_model.cuda()
    with torch.no_grad():
        for images, labels in tqdm(test_gen, total=len(test_gen)):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = loaded_model(images)
            outputs = nn.Sigmoid()(outputs)
            all_labels.append(labels.cpu())
            all_preds.append(outputs.cpu())

    prec, rec = calc_pres_recall(torch.cat(all_labels), torch.cat(all_preds))

    return all_labels, all_preds, prec, rec


class ImageFeaturesDataSet(Dataset):
    def __init__(self, df, y_cols, imagedir, x_col='filename', transform=None):
        self.df = df
        self.y_cols = y_cols
        self.imagedir = imagedir
        self.x_col = x_col
        self.df.reset_index(inplace=True, drop=True)
        self.filenames = list(df[x_col].values)
        self.idx = 0
        self.transform = transform

    def __getitem__(self, idx):
        im = io.imread(os.path.join(self.imagedir, self.filenames[idx]))
        im = im / 255.
        # sample = (im, self.labels[idx])
        if self.transform:
            # to avoid the type mismatch error. ToTensor() expects a float32
            im = im.astype(np.float32)
            im = self.transform(im)
        sample = (im, torch.tensor(self.df.iloc[idx][self.y_cols].values.astype(np.long)).type(torch.float32))
        return sample

    def __len__(self):
        return len(self.filenames)


def create_model(args, num_features):
    resnet = torchvision.models.resnet50(pretrained=True, progress=False)
    resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_features)
    return resnet


def main(args):
    # print("inside main")

    training_df = pd.read_csv('attributes_training_df.csv', index_col='index')
    testing_df = pd.read_csv('attributes_testing_df.csv', index_col='index')
    with open('all_features.pickle', 'rb') as f:
        all_features = pickle.load(f)
    num_features = len(all_features)
    train_gen, val_gen = get_training_gen(training_df, all_features)

    model = create_model(args, num_features)
    history = train(args, model, train_gen, val_gen)

    with open(f"{args.model_name}_history.pickle", 'wb') as f:
        pickle.dump(history, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="feature_detection", required=True)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    args = parser.parse_args()
    # Create a directory
    if not os.path.isdir(args.model_name):
        os.mkdir(args.model_name)
    logger = configure_logger(args)

    """
    Usage:
    ./featuredetectiontorch.py --model-name feature_detection --epochs 10 --lr 0.00001
    
    """

    main(args)
