# MLOps
Repo for Machine Learning Operations Course


Project 1 and 2 make a whole.

Project 1 involved training a model using Pytorch Lightning, with a training interface like MLFlow and hyperparameter tuning using Optuna.
Turn on mlflow using:

```bash
mlflow ui --port 5000
```

To train a model simply run:

```bash
py ./train.py
```
Project 2 involved using some inferencing server to be able to use a model using a REST api. Here the used framework was LitServe
To run project 2 server:

```bash
lightning deploy server.py
```
