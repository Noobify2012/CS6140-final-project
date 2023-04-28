# CS6140-final-project

Authors: Daniel Blum, Matthew Greene, Bram Ziltzer

Summary:  A classification project using Logistic Regression, SVMs, and DNNs to predict if a flight will be delayed or not.

# Running the project

### Perquisites
To run, you must have the raw data in the top level project directory.
[Use this kaggle link to download the raw data][Kaggle raw], and place it in a top level project directory under `./raw`.

Also, the project assumes ***Python 3.11.\*.***
If you don't have this, I recommend installing [Pyenv] (for many, many reasons), and installing 3.11.2 as a local python for this project.

## Locally

### Poetry
The project is based on [Poetry], so if you have poetry installed, just run 
```bash
poetry install
```
and everything will dump into an appropriate virtual env for you to run. 
Just make sure you have the poetry virtual environment running.

If you've never done this, I recommend the following:
1. First, run 
```bash
poetry config virtualenvs.prefer-active-python true
```
This will make sure `.venv` files will be located in the project directory.

2. Running this is VS Code and install the [Python extension] and the [Jupyter extension].

If you want to use PyCharm or DataSpell, [reach out to me](mailto:blum.da@northeastern.edu)

### CLI
If you're in the poetry shell venv for this project, just run `poetry build` to give you access to the simple CLI wrapper. 
Running 
```bash
final-project -h
```
will give you a little helper to show how to get the code running.
It will output some new png files, saved models, and a json with some stats in various folders.

### Jupyter
If you have the [Jupyter extension] set up, you can run jupyter notebooks natively in VS code. 
With the poetry venv installed, open up a `*.ipynb` file and make sure to select the local `.venv` environment.


## Docker
If you don't feel like dealing with any of this BS, there's some included docker files.
Sor, you have to make sure you have [Docker] installed on your system.
Once installed, head to the top level project directory and run
```bash
docker compose up -d
```

After some installs, this will launch a python/poetry environment running a jupyter server.
Just go to `http://localhost:8889` to gain access.

<!-- Links -->
[Poetry]: https://python-poetry.org/docs/
[Pyenv]: https://github.com/pyenv/pyenv
[VSCode]: https://code.visualstudio.com/download
[Docker]: https://docs.docker.com/get-docker/
[Mamba]: https://mamba.readthedocs.io/en/latest/index.html
[DataSpell]: https://www.jetbrains.com/dataspell/

[Jupyter Extension]: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
[Python Extension]: https://marketplace.visualstudio.com/items?itemName=ms-python.python


[kaggle raw]: https://storage.googleapis.com/kaggle-data-sets/2529204/4295427/compressed/raw.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230428%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230428T033007Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4308a0436957b68670876805cb62dd0aa4346615c15f882ab9afff59312a712fa783e981a2d43fa15dcac03ce1081fe620e5f8a7b8091b9237bafe60663ba7b234fd803b713e67db3f03ecc4c5e9fe772b557e92c6143d7f3ec9801c76579a09c4f9fd78f68341d31171419ca27f85e3e0a12bc5e7f61f93e4bb447ca0f8d06ff8a517b64c366b262baf419ff9ce7bebc9a5415e8d952845e92c7a99a3811739e0e7b7dde308d8ca1a38b5fd5354008c277799c6335025ee8aaa06276f8cad35a26ab86a743502674392b06fd88282d90a806067867ff3235f1fcb05233167e0e57a20843e33e24a8e9193c5371bf20a188b7a1e0eb0c0453e675eff4d7800e5