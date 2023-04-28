# CS6140-final-project

Authors: Daniel Blum, Matthew Greene, Bram Ziltzer

Summary:  A classification project using Logistic Regression, SVMs, and DNNs to predict if a flight will be delayed or not.

Github: https://github.com/Noobify2012/CS6140-final-project

# Running the project

### Perquisites
To run, you must have the raw data in the top level project directory.
[Use this google drive linke][google raw] to download the data, unzip and place it in a top level project directory so you have a `./raw` with a series of files named `Flights_20**_*.csv` there.

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


[google raw]: https://drive.google.com/file/d/13r082QtVGkihHgCd5QwQ8Xmw2uB9WJsf/view?usp=sharing