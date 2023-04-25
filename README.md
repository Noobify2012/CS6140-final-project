# CS6140-final-project

Authors: Daniel Blum, Matthew Greene, Bram Ziltzer

Summary:  A classification project using Logistic Regression, SVMs, and DNNs to predict if a flight will be delayed or not.

# The Docker Python Environment
The project is running in a collection of docker images via `docker compose`.
The `jupyter-lab` image runs a Debian instance with [Mamba] as the environment manager.

## Running the Poetry Environment

> ### Prerequisites:
> - Docker
> - Something to run Jupyter Notebook/Lab UI like [VSCode] or [DataSpell]

1. `cd` into the parent directory of the project
1. run `docker compose up -d`

Docker is going to do its thing for a bit and download a good amount of stuff.
Once done, you should have control over the terminal (as the `-d` is the detached option for `docker compose`).

If you run `docker ps`, you should have these running services:
- jupyter-lab
<!-- - flights-db
- pgadmin -->

<!-- You should also see two new volumes with `docker volumes ls`:
- cs6140-final-project_flights-db-volume
- cs6140-final-project_pgadmin-db-volume -->

And a new network with `docker network ls`:
- cs6140-final-project_cs6140-network

## Updating Poetry
If you need to add libraries 
TODO (dan) - FILL IN
<!-- ## Updating Mamba 
If you need to add libraries to the `jupyter-lab` service, it's easiest to do from inside the service:

1. Run `docker exec -it jupyter-lab /bin/bash` to get CLI access.
1. If not in the `app` folder, go ahead and `cd /app`.
1. Add the library you need with `mamba install -c conda-forge [library_name]`.
    > *note:* sometimes library names are different between *pip* and *conda-forge*, so its best to do a google search for the library with the term 'conda-forge' added for the name of the library if you can't find it.
1. Once installed, run `mamba env export > environment.yml` to "update" the environment file.

This update will persist during `docker compose up/down` operations but the `environment.yml` file will need to be updated in git. -->

<!-- ### Pulling a Mamba Update

If you're pulling a branch with an environment change, you want to force your image to rebuild.
Run `docker compose up -d --force-recreate --build jupyter-lab` -->

## Updating the PostgreSQL Image
<!-- TODO (dan) UNDER CONSTRUCTION -->
<!-- The Postgres image should be pretty self-contained. 
There is an attached docker volume which will persist between bringing the service up and down.
Not, this *is not* stored in git, so if you nuke the volume, you'll lose any changes you made to the DB that aren't committed to the relevant docker and sql files.

If you need to update the `create-db.sql` DDL script, you must also bring down the attached volume.

### If Docker is not running
run `docker compose down -v` - this will remove named volumes

### If Docker is running
run `docker compose rm -sfv flights-db` - this will forcefully shut down and remove the service and attached volumes.
Then, just `docker compose up -d` to relaunch all needed services.
 -->

# Running Jupyter

## VSCode
> ### Prerequisites
>
> - [Jupyter Extension] for VSCode
> - [Python Extension] for VSCode

When you open a `*.ipynb` file in VSCode, you'll want to the remote kernel running in docker.

1. In the top right corner of the `ipynb` file, select `Select Kernel` (if you don't see this, it might have another kernel already selected)l.
1. A drop-down pops up in the center of VSCode, select `Select Another Kernel`.
1. Select `Existing Jupyter Server...`
1. Enter `http://127.0.0.1:8889`
1. Select `Python 3 (ipykernel)`, the kernel at `/mamba/bin/python`

You should be all set!

## Browser
> ### Prerequisites
>
> - A Web Browser

Open up http://localhost:8889

## DataSpell
<!-- TODO (dan) UNDER CONSTRUCTION -->

# Running Poetry
*don't worry about this*
<!-- TODO (dan) UNDER CONSTRUCTION -->






<!-- Links -->
[VSCode]: https://code.visualstudio.com/download
[Poetry]: https://python-poetry.org/docs/
[Mamba]: https://mamba.readthedocs.io/en/latest/index.html
[DataSpell]: https://www.jetbrains.com/dataspell/

[Jupyter Extension]: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
[Python Extension]: https://marketplace.visualstudio.com/items?itemName=ms-python.python
