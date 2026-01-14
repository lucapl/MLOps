import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import lightning as L
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import os
import mlflow
import datetime


def accuracy(preds, labs):
    return torch.sum(torch.argmax(preds, dim=1) == labs)/len(preds)


class MnistClassifier(L.LightningModule):
    def __init__(self,
                 cnn_1_size=128, 
                 cnn_2_size=64,
                 cnn_3_size=32,
                 cnn_4_size=16, 
                 learning_rate=1e-3, ):
        super().__init__()
        self.save_hyperparameters()
        self.cnn_1 = nn.Conv2d(1, self.hparams.cnn_1_size, kernel_size=3, stride=1, padding=1)
        self.cnn_2 = nn.Conv2d(self.hparams.cnn_1_size, self.hparams.cnn_2_size, kernel_size=3, stride=1, padding=1)
        self.cnn_3 = nn.Conv2d(self.hparams.cnn_2_size, self.hparams.cnn_3_size, kernel_size=3, stride=1, padding=1)
        self.cnn_4 = nn.Conv2d(self.hparams.cnn_3_size, self.hparams.cnn_4_size, kernel_size=3, stride=1, padding=1)
        #self.linear_1 = nn.Linear(3*3*16, 64)
        self.linear_out = nn.Linear(3*3*self.hparams.cnn_4_size, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_function = nn.NLLLoss()

    def forward(self, x):
        x = self.relu(self.cnn_1(x))
        x = self.relu(self.maxpool(self.cnn_2(x)))
        x = self.relu(self.maxpool(self.cnn_3(x)))
        x = self.relu(self.maxpool(self.cnn_4(x)))
        x = x.view(x.size(0), -1)  # Flatten the input
        #x = self.linear_1(x)
        x = self.log_softmax(self.linear_out(x))
        return x

    def training_step(self, batch):
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss_function(preds, labels)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy(preds, labels))
        return loss

    def validation_step(self, batch):
        images, labels = batch
        preds = self.forward(images)
        # print(preds.shape)
        # print(labels.shape)
        loss = self.loss_function(preds, labels)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy(preds, labels))
        return loss

    def test_step(self, batch):
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss_function(preds, labels)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy(preds, labels))
        return loss 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MnistDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.data_folder = "./data"

    def prepare_data(self):
        MNIST(root=self.data_folder, train=True, download=True)
        MNIST(root=self.data_folder, train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(root=self.data_folder, train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [0.9, 0.1])
        self.mnist_test = MNIST(root=self.data_folder, train=False, transform=transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)


data = MnistDataModule(512)

def objective(trial):
    # Set the hyperparameters to optimize
    cnn_1_size = trial.suggest_int('cnn_1_size', 128, 256)
    cnn_2_size = trial.suggest_int('cnn_2_size', 64, 128)
    cnn_3_size = trial.suggest_int('cnn_3_size', 32, 64)
    cnn_4_size = trial.suggest_int('cnn_4_size', 16, 32)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)


    model = MnistClassifier(
        cnn_1_size,
        cnn_2_size,
        cnn_3_size,
        cnn_4_size,
        learning_rate=learning_rate
    )

    logger = MLFlowLogger(experiment_name="lightning_logs", run_name=f"optuna_logs/trial_{trial.number}", tracking_uri=mlflow.get_tracking_uri())
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')

    trainer = L.Trainer(
        max_epochs=5,
        callbacks=[pruning_callback],
        logger=logger,
        enable_progress_bar=False,  
        enable_model_summary=False  
    )

    trainer.fit(model, data)

    return trainer.callback_metrics['val_loss'].item()


def run_optimization(n_trials=5):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return study


if __name__ == '__main__':
    mlflowtracking_uri = "http://127.0.0.1:5000/"
    mlflow.set_tracking_uri(mlflowtracking_uri)
    study = run_optimization(n_trials=10)
    best_params = study.best_trial.params
    model = MnistClassifier(
        cnn_1_size=best_params['cnn_1_size'],
        cnn_2_size=best_params['cnn_2_size'],
        cnn_3_size=best_params['cnn_3_size'],
        cnn_4_size=best_params['cnn_4_size'],
        learning_rate=best_params['learning_rate']
    )
    #model = MnistClassifier()
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", run_name="final_training", tracking_uri=mlflow.get_tracking_uri())
    checkpointing = ModelCheckpoint(monitor="val_loss", dirpath="./model/") 
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices="auto",
        logger=mlf_logger,
        callbacks=[checkpointing],)
    #data = MnistDataModule(batch_size=512)
    trainer.fit(model, datamodule=data)
    results = trainer.test(model, datamodule=data)
    print("Best model at:", checkpointing.best_model_path)
    filepath = f"model_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.onnx"
    input_sample = torch.randn((1, 1, 28, 28))
    model.to_onnx(filepath, input_sample, export_params=True)
