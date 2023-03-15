# CS6140-final-project

Authors: Daniel Blum, Matthew Greene, Bram Ziltzer

Summary:  A classification project using Logistic Regression, SVMs, and DNNs to predict if a flight will be delayed or not.

> ### Prerequisites:
> - Docker
> - Something to run Jupyter Notebook/Lab UI like [VSCode] or a web browser


# Running Jupyter

This project sets up [Jupyter Lab][Jupyter] to run in a docker container in a mamba environment. 
Thus, any client that can run Jupyter UI ([VSCode], a web browser, etc.), can get local access to the Jupyter server.
***NOTE THAT SECURE ACCESS IS TURNED OFF SINCE IT'S ASSUMED ONLY LOCAL CONNECTIONS ARE MADE!!!***

> Before accessing, you must run `docker compose up -d` to launch the Jupyter server.
> For additional info, check out the environment section below.

### Accessing from The Web
Just navigate to http://localhost:8888 to get access to [Jupyter Lab][Jupyter]'s web UI.

### Accessing from VSCode
1. Open up a `.ipynb` file.
1. In the top right of [VSCode], you'll either see 'Select Kernel' or 'Python 3 (ipykernel)' if you've already established a connection;
click it.
1. A pop-down shows up, select 'Existing Jupyter Server...'
1. You might see multiple options if you've connected before. I recommend removing it by clicking the trash icon that appears when you hover over the server's name. 
Then click 'Enter the URL of the running Jupyter server'.
1. Enter either 'http://localhost:8888' or 'http://127.0.0.1:8888'.
***NOTE:*** Sometimes this shit is... finky.
If I get an error typing one address, I usually am successful when entering the other.

If you bring down the docker instance `jupyter-lab`, you might need to go through the process again.
I haven't fully figured out the eccentricities of it, so have fun :).


# The Environment

The project is running in a collection of docker images via `docker compose`.
The `jupyter-lab` image runs a Debian instance with [Mamba] as the environment manager.

## Running the Environment
1. `cd` into the parent directory of the project
1. run `docker compose up -d`

Docker is going to do its thing for a bit and download a good amount of stuff.
Once done, you should have control over the terminal (as the `-d` is the detached option for `docker compose`).

If you run `docker ps`, you should have two running services:
- jupyter-lab
- flights-db

## Updating the Mamba Environment
If you need to add libraries to the `jupyter-lab` service, it's easiest to do from inside the service:

1. Run `docker exec -it jupyter-lab /bin/bash` to get the CLI for the service.
1. If not in the `app` folder, go ahead and `cd /app`.
1. Add the library you need with `mamba install -c conda-forge [library_name]`.
    > *note:* sometimes library names are different between pip and conda/mamba, so its best to do a google search for the library with the term 'conda-forge' added.
1. Once installed, run `mamba env export > environment.yml` to "update" the environment file.

This update will persist during docker compose up/down operations.
If you remove the docker image for `jupyter-lab`, since you've updated the `environment.yml` file that's part of the repo, and new image creation will cause docker to rebuild the image now with the new library.

Don't forget to `git add environment.yml` and push if you want others to have the library as well :)


# PostgreSQL

## Updating the PostgreSQL Image
The Postgres image should be pretty self-contained. 
There is an attached docker volume which will persist between bringing the service up and down.
Not, this *is not* stored in git, so if you nuke the volume, you'll lose any changes made to the DB that are not committed to the relevant docker and SQL files.

If you need to update the  DDL script loaded from `Dockerfile-postgres`, you must remove the docker image and take down the volume.

1. run `docker compose down -v`
1. run `docker rmi cs6140-final-project-flights-db`
1. run `docker compose up -d`



<!-- Links -->
[Jupyter]: https://jupyter.org/
[VSCode]: https://code.visualstudio.com/download
[Mamba]: https://mamba.readthedocs.io/en/latest/index.html