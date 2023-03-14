# CS6140-final-project

Authors: Daniel Blum, Matthew Greene, Bram Ziltzer

Summary:  A classification project using Logistic Regression, SVMs, and DNNs to predict if a flight will be delayed or not.

# The Environment
The project is running in a collection of docker images via docker compose.
The `jupyter-lab` image runs a Debian instance with [Mamba] as the environment manager.

## Running the Environment

> ### Prerequisites:
> - Docker
> - Something to run Jupyter Notebook/Lab UI like [VSCode]

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

## Updating the PostgreSQL Iimage
The Postgres image should be pretty self-contained. 
There is an attached docker volume which will persist between bringing the service up and down.
Not, this *is not* stored in git, so if you nuke the volume, you'll lose any changes you made to the DB that aren't committed to the relevant docker and sql files.

If you need to update the `create-db.sql` DDL script, you must also bring down the attached volume.

### If Docker is not running
run `docker compose down -v` - this will remove named volumes

### If Docker is running
run `docker compose rm -sfv flights-db` - this will forcefully shut down and remove the service and attached volumes.
Then, just `docker compose up -d` to relaunch all needed services.

# Running Jupyter



<!-- Links -->
[VSCode]: https://code.visualstudio.com/download
[Mamba]: https://mamba.readthedocs.io/en/latest/index.html