# Tabata
Tabata est un package qui permet la manipulation de séries de signaux numériques.
C'est essentiellement une toolbox que je développe pour illuster une partie de mes cours de statistique descriptive. 
Notamment j'y laisse les sujets des examens de ces dernières années.
On y trouve un peu de machine-learning, un peu de visualisation graphique et beaucoup d'interactivité dans les notebooks Python.

    tabata
        + notebooks
        |    + opset_doc + instants_doc + tubes_doc + plots_doc
        |    + data
        |         + in (exemple de données)
        |         |   + AFL1EB.h5 (Aircraft FLight 1 Extended & Banalized)
        |         |
        |         + out (données produites par les notebooks
        |    + exercices
        |         + examen 20## *
        |         |   ...
        + scripts
        |    + pip_intall_all.bat (installation des packages utiles)
        |    + pipupdate.bat
        |    + jupyterlab_plotly_install.bat (plaotly sous jupyterlab)
        + opset.py
        |    + Opset
        |    + OpsetError
        + instants.py
        |    + Selector + indicator()
        + tubes.py
        |    + Tube + highlight() + AppTube
        + plots.py
             + nameunit() + selplot() + byunitplot() + get_colname()
             + groupplot() + doubleplot()

La plupart des analyses de données travaillent sur un tableau de mesures. Pourtant très souvent on a affaire à une liste de signaux. C'est le cas dans l'aéronautique quand on traite une série de vols (ou d'essais) et que chaque vol remonte un tableau de mesures indexé par le temps, souvent à une fréquence moyenne entre 1 Hz et 100 Hz. On a exactement la même chose quand on veut suivre les données d'usinage issues d'une machine-outil. Dans ce second cas, chaque pièce usinée donne un signal de mesures faites par la machine durant l'opération de production.

La première chose à faire quand on dispose de tels listes de signaux et de pouvoir les manipuler et les afficher. L'opset est un raccourci pour "liste d'opérations". Avec l'objet `Opset` il est facile de placer ses signaux stockés dans des DataFrames pandas dans un unique fichier HDF5. L'Opset réfère alors ce fichier et offre des fonctions d'itération et de visualisation.

Le sous-package "instants" contient l'objet `Selector` qui peremt de créer de manière interactive un détecteur d'instants spécifiques. Ce type d'instant correspond à des éléments graphiques visuels qu'un expert est capable d'identifier à l'écran. Le code utilise cet a priori pour construire une règle de décision très simple qui mime le comportement de l'expert.

Le sous-package "tubes" gère les opérations de scoring et de détection de signaux faible à l'aide de tubes de confiances adaptatifs.

**Un notebook spécifique détaille chacun de ces packages.**


Les tubes sont encore très rudimentaires, il reste à faire notamment une estimation des bornes statistiquement robuste à partir de valeurs extrêmes et pouvoir transférer un miodèle de tube sur un signal qui n'a pas servi à l'apprentissage. Je mets la plupart de ces choses "à faire" dans les _issues_.


## Installation/Requirements.

Pour faciliter une installation fonctionnelle, j'ai ajouté deux fichiers requirements que vous pouvez utiliser avec conda ou pip respectivement :

    conda create --name <env> --file requirements_conda.txt

ou

    python3 -m venv env
    source env/bin/activate 
    pip install -r requirements.txt

Pour tables qui est nécessaire au fonctionnement de pandas avec hdf5, si vous avez un problème, vous pouvez essayer ceci :

    pip install git+https://github.com/PyTables/PyTables.git@develop#egg=tables

Finalement, sous Windows, il peut rester quelques problèmes, auquel cas, on passe par un raccourci. Vous télécharger le package depuis https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables et vous l'installez avec wheel. Par exemple pour Python 3.7 :

    pip install tables‑3.6.1‑cp37‑cp37m‑win_amd64.whl

### Note pour MACOS + Git

Les denières versions de GitHub nécessitent un buffer assez large pour pouvoir stocker les images plotly.
Utiliser la commande 

    git config --global http.postBuffer 157286400

Pour grossir le buffer si vous obtenez le message d'erreur suivant :

    [info] error: RPC failed; HTTP 400 curl 22 The requested URL returned error: 400

(voir [https://](https://thewayeye.net/posts/fixing-the-rpc-failed-http-400-error-in-git/))