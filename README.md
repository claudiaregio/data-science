# Titanic Notebook

This notebook uses machine learning algorithms to get the best accuracy of predictions for who survived the sinking of the Titanic given the attributes in the dataset.

## Setup

- Install [Anaconda](https://www.anaconda.com/)
- Set the channel priority to strict to avoid issues with the environment creation taking forever.
  - `conda config --set channel_priority strict`
- Run the following commands (in either the terminal or an Anaconda Prompt):
  - `conda env create -f environment.yml`
  - `conda activate golden_scenario_env`
- In VS Code, open the [Titanic.ipynb](Titanic.ipynb) file and connect to the golden_scenario_env kernel

Also if you want to support PDF export from jupyter you need to setup LaTeX:

`sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic`

## Dev Containers

You can also run the notebooks inside dev containers:

* [![Open in Visual Studio Code](https://img.shields.io/static/v1?label=&message=Open%20in%20Visual%20Studio%20Code&color=blue&logo=visualstudiocode&style=flat)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/claudiaregio/data-science)
* [![Open in Github Codespaces](https://img.shields.io/static/v1?label=&message=Open%20in%20Github%20Codespaces&color=2f362d&logo=github)](https://codespaces.new/claudiaregio/data-science?quickstart=1&hide_repo_select=true)
