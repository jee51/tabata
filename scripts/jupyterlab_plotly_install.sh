# Installation de plotly sous jupyterlab
# Avec CONDA

# ---------------------------------
# Installatopn de NodeJS sous CONDA
# ---------------------------------

# Attention il ne sert à rien de dowloader une version de base de nodejs.
# Il faut installer sous l'environnement conda.

# https://anaconda.org/conda-forge/nodejs

conda install -c conda-forge/label/cf202003 nodejs

# ----------------------------------
# Installation de l'extention Plotly
# ----------------------------------

# https://anaconda.org/conda-forge/jupyterlab-plotly-extension

conda install -c conda-forge/label/cf202003 jupyterlab-plotly-extension

# --------------------------------------
# Installation de l'extension ipywidgets
# --------------------------------------

# https://discourse.jupyter.org/t/ipywidgets-for-jupyterlab-1-0/1675

jupyter labextension install @jupyter-widgets/jupyterlab-manager
