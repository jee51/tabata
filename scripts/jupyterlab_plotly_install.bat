rem Avoid "JavaScript heap out of memory" errors during extension installation
rem (OS X/Linux)
rem export NODE_OPTIONS=--max-old-space-size=4096
rem (Windows)
set NODE_OPTIONS=--max-old-space-size=4096

rem Jupyter widgets extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build

rem jupyterlab renderer support
jupyter labextension install jupyterlab-plotly --no-build

rem FigureWidget support
jupyter labextension install plotlywidget --no-build

rem Build extensions (must be done to activate extensions since --no-build is used above)
jupyter lab build

rem Unset NODE_OPTIONS environment variable
rem (OS X/Linux)
rem unset NODE_OPTIONS
rem (Windows)
set NODE_OPTIONS=