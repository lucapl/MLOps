# MLOps
Łukasz Andryszewski
Repo for Machine Learning Operations Course


## Project 1

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

## Project 2

Project 2 involved using some inferencing server to be able to use a model using a REST api. Here the used framework was LitServe
To run project 2 server:

```bash
lightning deploy server.py
```
## Project 3

Project 3 was done together with Jędrzej Pacanowski and involved deploying your model on an AWS instance in a Docker container.

On the AWS instance, install docker and clone the repo.

Copy the dockerfile in project3 then run the commands:

```bash
time docker build -t litserve-onnx-app:latest .
docker run -p 8000:8000 -it --entrypoint 'bash' --name dziala-projekt-3  litserve-onnx-app:latest
uv run server.py
INFO:     Uvicorn running on http://0.0.0.0:8000
```

After succesfully opening a TCP port 8000 in the AWS instance, the server can be tested with:

```bash
cd project3
python3 request.py
```
