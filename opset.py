# -*- coding: utf-8 -*-
"""
OPSET - Gestion des liste d'observations stockées dans des fichiers au format HDF5.

Chaque observation est un dataFrame pandas. Elle est nommée, le nom de l'enregistrement
est stocké dans la liste mais aussi comme nom de l'index temporel ce qui facilite
la recherche.

* L'itérateur `iterator()` renvoie un itérateur sur les observations du fichier HDF5.
* L'indexation de l'Opset renvoie un DataFrame, et il est plus simple d'itérer sur un slice.
* L'Opset est une classe permettant de visualiser interactivement le contenu.

**Versions**

1.0.3 - Algorithmes d'instants.
1.0.4 - Indextion négative et itération avec retour à l'origine.
1.0.5 - Réintégration des plots de DataFrames dans plots.py.
1.0.6 - Correction d'un bug dans put().


Created on Wed May  9 16:50:34 2018

@author: Jérôme Lacaille
"""

__date__ = "2026-01-26"
__version__ = '1.0.6'

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from .plots import nameunit, get_colname

###########################################################################
class OpsetError(ValueError):
    def __init__(self,filename,message):
        self.filename = filename
        self.message = message
        
    def __str__(self):
        return """Opset({self.filename})
    {self.message}""".format(self=self)
    
    

###########################################################################
#%% Un affichage interactif permettant de visualiser le contenu de la liste.
class Opset:
    """ Un affichage interactif utilisant plotly qui permet de
        visualiser les signaux stockés sous la formle de DataFrames pandas
        dans un fichier HDF5.
        
        Il s'agit de la classe principale de ce module. Elle permet de 
        charger un fichier HDF5 contenant une série de signaux et des les
        afficher un à un.
        
        Cette classe est construite pour être facilement dérivable de sorte
        que l'on puisse créer de nouvelles interfaces similaires pour différents
        algorithmes interactifs.

        :var storename:le nom du fichier HDF5.
        :var sigpos:   la position du pointeur (à partir de 0) dans le fichier.
        :var records:  la liste des noms d'enregistrements.
        :var colname:  la variable à afficher par défaut.
        :var phase:    un nom de variable booléen (None par défaut) représentant
                        instants à mettre en évidence.
        :var df:       la DataFrame courant (celui qui correspond à `sigpos`.)
    """

    def __init__(self, storename, phase=None, pos=0, name=None, sortkey=None):
        """ Initialisation de l'Opset.
        
            Le constructeur de l'Opset nécessite un fichier HDF5, si le fichier
            transmis n'existe pas il sera créé pour éventuellement un remplissage par
            la méthode `put()`.

            :param storename: le datatore correspondant à l'OPSET.
            :param phase:     le nom d'une colonne binaire contenant les indices
                                d'une pase particulière à surligner en rouge.
            :param pos:       un numéro de signal à charger par défaut, 
                                renseigne la variable sigpos.
            :param name:      le début d'un nom de variable (colonne du signal)
                                par défaut (variable colname).
            :param sortkey:   l'ordre de lecture des enregitrements par 'sorted'.
        """
        
        if isinstance(storename,Opset):
            storename = storename.storename
            
        if not os.path.isfile(storename): # Il faut créer le fichier.
            newstore = pd.HDFStore(storename, mode='w')
            newstore.close()
            
        self.storename = storename
        with pd.HDFStore(self.storename, mode='r') as store:
            self.records = store.keys()
            if sortkey:
                self.records = sorted(self.records, key=sortkey)
            nbmax = len(self.records)
            if (pos < 0) or (pos >= nbmax):
                pos = 0
            if nbmax>0:
                self.df = store[self.records[pos]]
                colname = get_colname(self.df.columns,name)
                phase = get_colname(self.df.columns,phase,default=None)
            else:
                self.df = None
                colname = None
                phase = None
                
        self.sigpos = pos
        self.colname = colname
        self.phase = phase
        
        self._tmppos = 0 # Variable cachée pour l'itération.
        
    def __repr__(self):
        """ Affichage du nom de l'Opset et de la liste des instants selectionnés."""

        return "OPSET '{}' de {} signaux.\n\
        position courante : sigpos  = {}\n\
        variable courante : colname = {}\n\
        phase surlignée   : phase   = {}"\
            .format(self.storename, len(self.records),
                    self.sigpos, self.colname, self.phase)

    def __len__(self):
        """ La longueur de l'Opset."""
        return len(self.records)
        
    def __getitem__(self,pos):
        """ Récupère le DataFrame à la position souhaitée.
            
            Déplace le pointeur en conséquence.
        """
        
        if isinstance(pos,slice):
            ind = range(len(self.records))
            return self.iterator(ind[pos])
        
        if hasattr(pos, '__iter__'):
            return self.iterator(pos)
        
        if pos<=-len(self.records) or pos>= len(self.records):
            raise OpsetError(self.storename,
                            "La position doit être comprise entre {} \
                            et {}".format(-len(self.records),len(self.records)-1))
    
        if pos<0:
            pos = len(self.records)+pos
        
        if pos != self.sigpos:
            self.sigpos = pos
            rec = self.records[pos]
            self.df = pd.read_hdf(self.storename, rec)
            
        return self.df
        
    
    def iterator(self, *argv):
        """ Itération sur les éléments du HDF5.
        
            Il y a plusieurs façons d'appeler l'itérateur :
        
            * iterator()           : itère sur tous les fichiers.
            * iterator(nb)         : itère sur les nb premiers fichiers.
            * iterator(first,last) : itère sur une partie des fichiers 
                                     (first inclu, pas last)
            * iterator(iterable)   : itère sur une liste.
        
            :param first: le premier élément  itérer (commence à 0) ou un itérable.
            :param last:  le dernier élément (la fin du fichier si non précisé)
        
            :return:      le DataFrame à chaque itération.
        """
        
        if len(argv)==0:
            seq = range(len(self.records))
        elif len(argv)==1 and hasattr(argv[0], '__iter__'):
            seq = argv[0]
        else:
            seq = range(*argv)
            
        _tmppos = self.sigpos
        
        for i in seq:
            yield self[i]
    
        self[_tmppos]
    
    def rewind(self,sigpos=0):
        """ Réinitialisation du pointeur au début du fichier.
        
            Cette méthode renvoie l'Opset ce qui fait qu'on peut 
            enchainer les appels.
        """
        self[sigpos]
        return self
    
    def __iter__(self):
        return self.iterator()
    
    def current_record(self):
        """ Renvoie le nom de l'enregistrement courant."""
        if len(self.records)==0:
            raise OpsetError(self.storename, "Opset is empty.")
        else:
            return self.records[self.sigpos]
 

    def clean(self):
        """ Réinitialise le fichier.
        
            Cette méthode renvoie l'Opset ce qui fait qu'on peut 
            enchainer les appels.
        """
        
        if os.path.exists(self.storename):
            os.remove(self.storename)
        self.__init__(self.storename)
                      
        return self
        
        
    def put(self,df,record=None):
        """ Stocke le signal dans le fichier.
        
            :param df:     le signal à stocker.
            :param record: l'enregistrement du signal 
 
            Si aucun nom d'enregistrement n'est donné on regarde df.index.name.
            Si le nom de l'enregistrement n'existe pas il est rajouté.
            
            L'Opset est alors positionné sur cet enregistrement.
        """

        if record is None:
            if (not df.index.name) or len(df.index.name)==0:
                raise OpsetError(self.storename, "Record name is missing.")
            record = '/' + df.index.name

        if record in self.records:
            self.sigpos = self.records.index(record)
        else:
            self.sigpos = len(self.records)
            self.records.append(record)

        if (not df.index.name) or len(df.index.name)==0:
            df.index.name = record
        
        # Enregistrement du DataFrame.
        self.df = df
        df.to_hdf(self.storename,key=record)
        
        # Si l'Opset était vide maintenant il faut spécifier une variable.
        self.colname = df.columns[0]


    # ------------------------ Affichage ----------------------------------
    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Crée l'affichage interactif des courbes.
        
            Cette fonction définit les différents éléments de l'affichage. 

            :param f:       un pointeur sur la figure à créer.
            :param phase:   le nom d'une colonne binaire contenant les
                            points à mettre en évidence si besoin.
            :param pos:     le numéro du signal à afficher en premier sinon
                            le premier signal du fichier.
            :param name:    le nom de la variable à afficher en premier
                            sinon la premiere variable du premier signal.

            On la décompose de la fonction d'affichage pour avoir plus de
            flexibilité si on souhaite dériver la classe et proposer
            d'autres interfaces graphiques. Une méthode principale `plot()` doit
            d'abord créer la figure avec `make_subplot()` même si ici on utilise
            qu'un unique graphe. Cette figure est ensuite passée en agument `f`.

            Cette version crée 5 objets à l'écran :

            * `variable_dropdown`:  une liste de variables.
            * `signal_slider`:      la scrollbar correspondant aux
                                    différents signaux.
            * `previous_button`:    le bouton 'précédent'.
            * `next_button`:        le bouton 'suivant'.
            * `update_function`:    la fonction de mise à jour de 
                                    l'affichage.

            La mise à jour de l'affichage se fait par le callback 
            `update_function()`. Dans la méthode `plot()` elle est exécutée
            par l'appel à la fonction `interactive()`:

              out =  widgets.interactive(update_function,
                                        colname=variable_dropdown,
                                        sigpos=signal_slider)

            Si l'on modifie cette fonction il faudra d'abord stocker sa valeur
            puis appeler l'appeler avec les objets correspondants si on souhaite
            qu'ils restent actifs.

            :return: le dictionnaire d'éléments utiles à l'affichage décrit
                        ci-dessus.
        """

        # Une erreur s'il n'y a rien dans le fichier.
        nbmax = len(self.records)
        if nbmax==0:
            raise OpsetError(self.storename, "Opset is empty.")
        
        if (pos is not None) and (pos >= 0) and (pos < nbmax) and (pos != self.sigpos):
            self.sigpos = pos
        # On relit systématiquement le fichier au début au cas où une nouvelle colonne
        # serait ajoutée.
        self.df = pd.read_hdf(self.storename,self.records[self.sigpos])

        if (name is not None):
            self.colname = get_colname(self.df.columns,name)
        if (phase is not None):
            self.phase = get_colname(self.df.columns,phase,default=None)
        
        # Définition des données à afficher.
        f.add_trace(go.Scatter(x=self.df.index, y=self.df[self.colname],name="value"),
                    row=1,col=1)
        if self.phase:
            ind = self.df[self.phase]
            f.add_trace(go.Scatter(x=self.df.index[ind], 
                                   y=self.df[self.colname][ind],
                                   name="phase",
                                   line={'color':'red'}),
                       row=1, col=1)
        
        # Description de la figure graphique.
        f.update_layout(width=500, height=400, showlegend=False)
      
        
        # ---- Callback de mise à jour de la figure ----
        def update_plot(colname, sigpos):
            """ La fonction d'interactivité avec les gadgets.
            
                Ce callback utilise la continuation de la figure `f`. 
                On fait cela pour qu'à chaque appel une nouvelle figure
                soit effectivement créée et empilée dans le notebook.
            """
            
            self.colname = colname
            name, unit = nameunit(colname)
            
            # Pour éviter de relire deux fois le signal initial.
            if sigpos != self.sigpos:
                self.sigpos = sigpos
                self.df = pd.read_hdf(self.storename, self.records[sigpos])

            # Mise à jour des courbes.
            f.update_traces(selector=dict(name="value"),
                            x = self.df.index, y = self.df[self.colname])
            # Affichage superposé de la phase identifiée.
            if self.phase is not None:
                ind = self.df[self.phase]
                f.update_traces(selector=dict(name="phase"),
                                x = self.df.index[ind],
                                y = self.df[self.colname][ind])

            # Mise à jour des titres et labels.
            f.update_layout(title=self.df.index.name, 
                            yaxis_title=name + '  [ ' + unit + ' ]')
        # ---- Fin du calback ----
         
            
        # Construction des gadgets interactifs.
        wd = widgets.Dropdown(options=self.df.columns,
                              value=self.colname,
                              description="Variable :")
        wbp = widgets.Button(description='Previous')
        wbn = widgets.Button(description='Next')
        ws = widgets.IntSlider(value=self.sigpos, min=0, max=nbmax-1, step=1,
                               orientation='vertical',
                               description='Record',
                               continuous_update=False,
                               layout=widgets.Layout(height='360px'))


        # ---- Callback des boutons ----
        def wb_on_click(b):
            """ Callbacks des boutons Previous et Next."""
            if b.description == 'Previous':
                if ws.value > 0:
                    ws.value -= 1
            if b.description == 'Next':
                if ws.value < ws.max:
                    ws.value += 1
        # ---- Fin du callback ----

        wbp.on_click(wb_on_click)
        wbn.on_click(wb_on_click)
        

        # Mise à jour de l'affichage.
        update_plot(self.colname, self.sigpos)
        
        # On renvoie le dictionnaire des objets graphiques.
        return dict(variable_dropdown = wd,
                    signal_slider = ws,
                    previous_button = wbp,
                    next_button = wbn,
                    update_function = update_plot)
    
    
    def plot(self,phase=None,pos=None,name=None):
        """ Affichage de l'interface.
        
            La méthode `plot()` commence par créer les différents éléments par
            un passage de ses paramètres à `make_figure()`, puis elle doit
            mettre en oeuvre l'interactivité par un appel à `interactive`
            et construire le look de l'interface en positionnant les 
            objets. Il est aussi possible de modifier le `layout`de la 
            figure.

            En entrée, les mêmes paramètres que `make_figure()`,
            et en sortie une organisation des éléments dans une boite.
            
            Il est important de créer la figure avec `make_subplots()` car
            on pourra ainsi utiliser les position des graphes dans le cas d'une
            dérivation de la classe.
        """

        f = make_subplots(rows=1, cols=1)
        f = go.FigureWidget(f)
        e = self.make_figure(f, phase,pos,name)
        out = widgets.interactive(e['update_function'], 
                                  colname=e['variable_dropdown'], 
                                  sigpos=e['signal_slider'])
        boxes = widgets.VBox([widgets.HBox([e['variable_dropdown'], 
                                            e['previous_button'], 
                                            e['next_button']]),
                              widgets.HBox([f, e['signal_slider']])])
        
        return boxes
    
    def plotc(self,phase=None,pos=None,name=None):
        """ Affichage de l'interface sans passage par FigureWidgets."""
        f = make_subplots(rows=1, cols=1)
        #f = go.FigureWidget(f)
        e = self.make_figure(f, phase,pos,name)
        
        def update_plot_c(colname, sigpos):
            e['update_function'](colname, sigpos)
            f.show()
        
        out = widgets.interactive_output(update_plot_c, dict(
                                  colname=e['variable_dropdown'], 
                                  sigpos=e['signal_slider']))
        boxes = widgets.VBox([widgets.HBox([e['variable_dropdown'], 
                                            e['previous_button'], 
                                            e['next_button']]),
                              widgets.HBox([out, e['signal_slider']])])
        
        return boxes
    
###########################################################################
#%% Récupération d'un jeu d'exemples.
def datafile(name=''):
    filename = os.path.join(os.path.dirname(__file__),'notebooks/data',name)
    return filename
