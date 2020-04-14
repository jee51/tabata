# Tabata
Tabata est un package qui permet la manipulation de séries de signaux numériques.

  tabata
    |
    + opset
    |    + Opset
    |
    + etdex.py
    |    + Etdex
    |
    + banalyse.py
    |    + banalise
    |
    + instants
    |    + Selector
    |
    + tubes
         + Tube

La plupart des analyses de données travaillent sur un tableau de mesures. Pourtant très souvent on a affaire à une liste de signaux. C'est le cas dans l'aéronautique quand on traite une série de vols (ou d'essais) et que chaque vol remonte un tableau de mesures indexé par le temps, souvent à une fréquence moyenne entre 1 Hz et 100 Hz. On a exactement la même chose quand on veut suivre les données d'usinage issues d'une machine-outil. Dans ce second cas, chaque pièce usinée donne un signal de mesures faites par la machine durant l'opération de production.

La première chose à faire quand on dispose de tels listes de signaux et de pouvoir les manipuler et les afficher. L'opset est un raccourci pour "liste d'opérations". Avec l'objet `Opset` il est facile de placer ses signaux stockés dans des DataFrames pandas dans un unique fichier HDF5. L'Opset réfère alors ce fichier et offre des fonctions d'itération et de visualisation.

L'objet `Etdex` est particulier à l'aéronautique française. C'est un code permettant de convertir des données issues de bancs d'essais et sauvegardées dans le format propriétaire ETDEX (un format de stockage numérique plat obsolète aujourd'hui) dans un DataFrame. Je laisse ce code dans le package Tabata car il est facile de s'en inspirer pour lire d'autres formats ressemblants.

La fonction `banalise` transforme un Opset en un Opset banalisé à laide de trois procédures : changer les noms des variables à l'aide d'un fichier d'alias, changer les dates des index, appliquer un facteur aléatoire (mais proche de 1) aux données numériques.

Le sous-package "instants" contient l'objet `Selector` qui peremt de créer de manière interactive un détecteur d'instants spécifiques. Ce type d'instant correspond à des éléments graphiques visuels qu'un expert est capable d'identifier à l'écran. Le code utilise cet a priori pour construire une règle de décision très simple qui mime le comportement de l'expert.

Le sous-package `Tubes` gère les opérations de scoring et de détection de signaux faible à l'aide de tubes de confiances adaptatifs.

## Fonctionnalités codées

- [x] Lecture de fichiers ETDEX - Le format ETDEX est un format plat permettant de stocker des données issues des bancs d'essais de moteurs turbofans. (Cette fonction ne gère pour l'instant que les données transitoires mono-fréquence.)
- [x] Fonction de banalisation - Transforme un fichier de signaux HDF5 en un nouveau fichier banalisé. Un fichier d'alias est produit et reste avec la source. Il peut être adapté par exemple pour changer les noms des variables.
- [x] Création d'opsets - Les opsets sont des listes de DataFrames pandas stockées dans des fichiers HDF5. Pour cela on acréé l'objet `Opset`.
- [x] Affichages de signaux - Les signaux sont affichés dans des notebooks avec une interactivité permettant de sélectionner les variables ou les enregistrements.
- [x] Un objet `Sélector` dérive de l'Opset, il permet de sélectionner interactivement des points d'intérêt sur un ou plusieurs signaux et génère un code permettant de retrouver ce point d'intérêt automatiquement. 
- [x] L'objet `Selector` permet aussi d'extraire une partie d'un signal à gauche ou a droite du point d'intéret ou entre deux instants.
- [ ] Un objet Tube permet de générer des tubes de confiance autour d'une ou plusieurs courbes.
 

