#!/usr/bin/env python3

import torch, torchvision
from torch import nn, functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms, Resize, Normalize

from skimage import io, transform

import os
import pathlib
import argparse
import logging.handlers
import pickle
import time
from tqdm import tqdm

from collections import OrderedDict

BATCH_SIZE = 32
LOG_DIR = "logs"
FORMAT = "[%(levelname)s] %(message)s"

DATASET_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/'
TRAIN_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_train_split/whole_resize'
TEST_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_eval_img_resize/'


class ImageDataset(Dataset):
    def __init__(self, category_dict, root_dir, transform=None):
        self.image_names = list(category_dict.keys())
        self.mapped_labels = category_dict
        self.labels = list(category_dict.values())
        if not os.path.isdir(root_dir):
            pass
            #raise FileNotFoundError("Directory not found")
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        im = io.imread(os.path.join(self.root_dir, self.image_names[idx]))
        im = im / 255.

        if self.transform:
            im = self.transform(im)
        sample = (im, torch.tensor(self.labels[idx]).type(torch.long))
        return sample


class ToTorchTensor(object):
    """Convert ndarrays in sample to Torch image Tensors. (CxWxH)"""

    def __call__(self, im):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1))
        return torch.from_numpy(im).type(torch.float32)


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


def create_training_datasets(args, samples=None):
    with open('training_mapped_categories.pickle', 'rb') as f:
        training_dict = pickle.load(f)

    if samples:
        training_dict = dict(list(training_dict.items())[:samples])
    training_transforms = transforms.Compose([ToTorchTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageDataset(training_dict, TRAIN_IMAGE_PATH, transform=training_transforms)
    data_len = len(dataset)
    val_split = 0.25
    val_len = int(val_split * data_len)
    train_len = data_len - val_len
    logger.info(f"The training dataset lengths: train: {train_len}, val: {val_len}")

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    return train_ds, val_ds


def create_testing_datasets(args, samples=None):
    with open('testing_mapped_categories.pickle', 'rb') as f:
        testing_dict = pickle.load(f)
    if samples:
        testing_dict = dict(list(testing_dict.items())[:samples])
    test_ds = ImageDataset(testing_dict, TEST_IMAGE_PATH, transform=ToTorchTensor())
    logger.info(f"The training dataset lengths: test: {test_ds}")
    return test_ds


def create_training_dataloaders(args):
    training_ds, validation_ds = create_training_datasets(args, samples=None)
    train_loader = DataLoader(training_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, drop_last=True)

    return train_loader, val_loader


def create_testing_dataloaders(args):
    testing_ds = create_testing_datasets(args, samples=None)
    test_loader = DataLoader(testing_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, drop_last=True)
    return test_loader


def is_cuda():
    return torch.cuda.is_available()


def get_device():
    if is_cuda():
        device_number = os.environ['CUDA_VISIBLE_DEVICES']
        device_number = 1 if not device_number else 0
        dev = f"cuda:{device_number}"
    else:
        dev = "cpu"

    device = torch.device(dev)

    return device


def _validate_model(args, model, validation_gen, loss_fn):
    val_loss = 0.0
    val_acc = 0.0
    # batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    total_validation_samples = BATCH_SIZE * len(validation_gen)
    model.eval()
    for images, labels in validation_gen:
        if is_cuda():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        val_loss += loss.cpu().item() * labels.size(0)
        acc = torch.sum(torch.max(outputs.data, 1).indices == labels.data)
        val_acc += acc.cpu().item()

    val_loss = val_loss / total_validation_samples
    val_acc = val_acc / total_validation_samples

    return val_loss, val_acc


def train_model(args, model):
    history = {"loss": [],
               "acc": [],
               "val_loss": [],
               "val_acc": []}
    lr = args.lr if args.lr else 0.0001
    # batch_size = args.batch_size if args.batch_size else BATCH_SIZE

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    training_gen, validation_gen = create_training_dataloaders(args)

    total_training_samples = BATCH_SIZE * len(training_gen)

    epochs = args.epochs
    device = get_device()
    logger.info(f"The current device is : {device}")
    if is_cuda():
        torch.cuda.set_device(device)
        model.cuda()

    for epoch in range(epochs):
        start = time.time()
        train_loss = 0.0
        train_acc = 0.0

        logger.info(f"Training epoch: {epoch+1}")
        model.train()
        for batch, (images, labels) in tqdm(enumerate(training_gen), desc='batch: ', total=len(training_gen)):
            if is_cuda():
                # May be we can get rid of Variable
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Flushes the accumulated gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(images)
            loss = cross_entropy_loss(outputs, labels)
            # Calls back propagation
            loss.backward()
            # Applies the Gradient to the parameters
            optimizer.step()
            # Loss calculated is the mean per batch.
            # Hence, we multiply with the batch size
            # .item() is required to fetch a scalar value from tensor
            train_loss += loss.cpu().item() * labels.size(0)
            # .data is used to fetch the tensor from the Variable
            # indices return the argmax
            acc = torch.sum(torch.max(outputs.data, 1).indices == labels.data)
            train_acc += acc.cpu().item()

        train_acc = train_acc/total_training_samples
        train_loss = train_loss/total_training_samples
        # no gradients are required for inference or evaluation
        with torch.no_grad():
            val_loss, val_acc = _validate_model(args, model, validation_gen, cross_entropy_loss)

        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        end = time.time()
        duration = end-start
        logger.info(f"epoch: {epoch+1}\t loss: {train_loss: 0.4f} acc: {train_acc: 0.3f} val_loss: {val_loss: 0.4f} val_acc: {val_acc: 0.3f} time: {duration: 0.2f} secs")

    model_fname = f"{args.model_name}-model.pickle"
    logger.info(f"Saving the model to : {model_fname}")
    torch.save(model, model_fname)
    return history


def create_model(args):
    num_categories = 68
    base_model = torchvision.models.resnet50(pretrained=True, progress=False)
    # Modifies the existing fc layer and adds an additional fc layer
    base_model.fc = torch.nn.Linear(base_model.fc.in_features, 1024)
    new_model = nn.Sequential(base_model, nn.ReLU(), nn.Linear(in_features=1024, out_features=num_categories))
    logger.info(new_model)

    return new_model


def main(args):
    logger.info("inside main")
    is_cuda = torch.cuda.is_available()
    logger.info(f"Is Cuda Available: {is_cuda}")

    model = create_model(args)
    history = train_model(args, model)
    logger.info(history)
    fname = os.path.join(args.model_name, 'history.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"history file saved to : {fname}")
    #logger.info(model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="image_classification", required=True)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    args = parser.parse_args()
    # Create a directory
    if not os.path.isdir(args.model_name):
        os.mkdir(args.model_name)
    logger = configure_logger(args)

    main(args)
