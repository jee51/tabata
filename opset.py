# -*- coding: utf-8 -*-
"""
OPSET - Gestion des liste d'observations stockées dans des fichiers au format HDF5.

Chaque observation est un dataFrame pandas. Elle est nommée, le nom de l'enregistrement
est stocké dans la liste mais aussi comme nom de l'index temporel ce qui facilite
la recherche.

* Les noms de variables sont composées d'un nom suivi de l'unité entre crochets "[]".
* La fonction auxiliaire nameunit décompose le nom d'une variable.
* L'itérateur Iterator renvoie un itérateur sur les observations du fichier HDF5.
* Le Selector est une classe permettant de visualiser interactivement le contenu.
* Le Selector permet aussi de sélectionner des points sur des courbes.


Created on Wed May  9 16:50:34 2018

@author: Jérôme Lacaille
"""

__date__ = "2020-03-30"
__version__ = '1.0.2'

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots

###########################################################################
class OpsetError(ValueError):
    def __init__(self,filename,message):
        self.filename = filename
        self.message = message
        
    def __str__(self):
        return """Opset({self.filename})
    {self.message}""".format(self=self)
    
    
###########################################################################
#%% Fonctions auxiliaires.
def nameunit(col,sep="["):
    """ Renvoie le nom et l'unité d'une variable au format NOM[UNITE]"""
    i = col.find(sep)
    if i == -1:
        return col,'-'
    return col[:i],col[i+1:-1]


def byunits(cols,sep="["):
    """ Récupère un dictionnaire dont les clés sont les unités uniques et
        les valeurs les colonnes correspondates.
    """
    dnu = dict()
    for col in cols:
        name,unit = nameunit(col,sep)
        if unit in dnu:
            dnu[unit].append(col)
        else:
            dnu[unit] = [col]
    return dnu


def get_colname(columns,variable,default=0):
    """ Récupère le nom complet de la variable.
    
        :param columns:  La liste des retours possibles.
        :param variable: Le début du nom rechercé dans la liste.
        :param default:  Un moyen de différentier si on veut une valeur 
                            ou rien quand rien n'est trouvé.
                            
        Par défaut la fonction renvoie la première donnée.
        En posant `Default=None` une entrée vide renvoie le vide.
    """
    
    if default is not None and isinstance(default,int):
        default = columns[default]
    if not variable:
        return default
    
    subs = [r for r in columns if variable in r]
    if len(subs) > 0:
        variable = subs[0]
    else:
        variable = default

    return variable


###########################################################################
#%% Fonctions d'affichage de signaux.
def selplot(df, variable=None):
    """ Affiche un signal parmis la liste des signaux disponibles.

        :param df:       la table de données.
        :param variable: une variable à afficher au lieu de la première colonne de la table.
    """

    def selected_plot(col):
        """ La fonction d'interactivité de `selplot`.
        
            C'est cette fonction qui définit notamment le style du titre et des axes.
        """
        name, unit = nameunit(col)
        data = [go.Scatter(x=df.index, y=df[col])]
        layout = go.Layout(title=name,
                           titlefont={'color': "blue"},
                           xaxis={'title': df.index.name,
                                  'titlefont': {'color': "blue"}},
                           yaxis={'title': unit,
                                  'titlefont': {'color': "blue"}})
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    variable = get_colname(list(df.columns),variable)
    wd = widgets.Dropdown(options=df.columns, value=variable, description="Variable :")
    widgets.interact(selected_plot, col=wd)


def byunitplot(df, yunit=None, xunit="date", title=""):
    """ Affiche les signaux en fonction de leur unité.

        Au début l'affichage est vide et une question est posée '?',
        en choisissant une unité on affiche toutes les courbes correspondant.
        On peut aussi sélectionner quelques courbes en cliquant sur les légendes.
        Un racourci permet de n'afficher qu'une seule courbe.

        :param df:    la table de données.
        :param title: un titre à la figure.
        :param xunit: l'unité de date.
        :param yunit: l'unité des observations.
    """
    dnu = byunits(df)
    units = list(dnu.keys())

    def unit_plot(unit, variable='All'):
        """ Fonction d'interactivité des gadgets."""
        if unit not in dnu:
            return
        if variable == 'All' or variable is None:
            cols = dnu[unit]
            data = [go.Scatter(x=df.index, y=df[col], name=nameunit(col)[0])
                    for col in cols]
        else:
            print(variable)
            data = [go.Scatter(x=df.index, y=df[variable],
                               name=nameunit(variable)[0])]
        layout = go.Layout(title=title,
                           titlefont={'color': "blue"},
                           xaxis={'title': xunit,
                                  'titlefont': {'color': "blue"}},
                           yaxis={'title': unit,
                                  'titlefont': {'color': "blue"}},
                           showlegend=True)
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def update_variables(*args):
        """ Fonction de mise à jour des listes déroulantes."""
        wv.options = ['All'] + dnu[wu.value]
        wv.value = 'All'

    if (yunit is None) or (yunit not in units):
        yunit = units[0]
    wu = widgets.Dropdown(options=units, value=yunit, description="Unité :")
    wv = widgets.Dropdown(options=['All']+dnu[units[0]], 
                          value='All', description="Variables :")
    #wu = widgets.Dropdown(options=["?"] + units, description="Unité :")
    #wv = widgets.Dropdown(options=['?'], description="Variables :")
    wu.layout = widgets.Layout(width='30%')
    wu.observe(update_variables, 'value')
    b = widgets.HBox([wu, wv])
    out = widgets.interactive_output(unit_plot, dict(unit=wu, variable=wv))
    return widgets.VBox([b,out])


###########################################################################
#%% Un affichage interactif permettant de visualiser le contenu de la liste.
class Opset:
    """ Un affichage interactif utilisant plotly qui permet de
        sélectionner des points dans une liste d'observations.
        
        Il s'agit de la classe principale de ce module. Elle permet de 
        charger un fichier HDF5 contenant une série de signaux et des les
        afficher un à un.
        
        Cette classe est construite pour être facilement dérivable de sorte
        que l'on puisse créer de nouvelles interfaces similaires pour différents
        algorithmes interactifs.
    """

    def __init__(self, storename, phase=None, pos=0, name=None):
        """ Initialisation de l'Opset.
        
            Le constructeur de l'Opset nécessite un fichier HDF5, si le fichier
            transmis n'existe pas il sera créé pour éventuellement un remplissage par
            la méthode `put`.

            :param storename: le datatore correspondant à l'OPSET.
            :param phase:     le nom d'une colonne binaire contenant les indices
                                d'une pase particulière à surligner en rouge.
            :param pos:       un numéro de signal à charger par défaut.
            :param name:      le début d'un nom de variable (colonne du signal)
                                par défaut.
        """
        
        if not os.path.isfile(storename): # Il faut créer le fichier.
            newstore = pd.HDFStore(storename, mode='w')
            newstore.close()
            
        self.storename = storename
        with pd.HDFStore(self.storename, mode='r') as store:
            self.records = store.keys()
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

        
        
    def __repr__(self):
        """ Affichage du nom de l'Opset et de la liste des instants selectionnés."""

        return "OPSET '{}' de {} signaux.\n\
                position courante : {}\n\
                variable : {}".format(self.storename, len(self.records),
                                      self.sigpos, self.colname)

    
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
        
            :return:      le DataFrame
        """
        
        if len(argv)==0:
            seq = range(len(self.records))
        elif len(argv)==1 and hasattr(argv[0], '__iter__'):
            seq = argv[0]
        else:
            seq = range(*argv)
            
        for i in seq:
            self.sigpos = i
            rec = self.records[i]
            self.df = pd.read_hdf(self.storename, rec)
            yield self.df
    
    
    def rewind(self,sigpos=0):
        for df in self.iterator(sigpos,sigpos+1):
            pass
    
    
    def current_record(self):
        """ Renvoie le nom de l'enregistrement courant."""
        if len(self.records)==0:
            raise OpsetError(self.storename, "Opset is empty.")
        else:
            return self.records[self.sigpos]
 

    def clean(self):
        """ Réinitialise le fichier."""
        
        if os.path.exists(self.storename):
            os.remove(self.storename)
        self.__init__(self.storename)
        
        
    def put(self,df,record=None):
        """ Stocke le signal dans le fichier.
        
            :param df:     le signal à stocker.
            :param record: l'enregistrement du signal 
            :param reset:  recrée un fichier vide.

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
        
        self.df = df
        df.to_hdf(self.storename,record)


    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Crée l'affichage interactif des courbes.
        
            Cette fonction définit les différents éléments de l'affichage. 

            :param phase:   le nom d'une colonne binaire contenant les
                            points à mettre en évidence si besoin.
            :param pos:     le numéro du signal à afficher en premier sinon
                            le premier signal du fichier.
            :param name:    le nom de la variable à afficher en premier
                            sinon la premiere variable du premier signal.

            On la décompose de la fonction `plot` pour avoir plus de
            flexibilité si on souhaite dériver la classe et proposer
            d'autres interfaces graphiques.

            Cette version crée 6 objets :

            * `figure`:             la figure contenat les axes où l'on 
                                    affiche la courbe.
            * `variable_dropdown`:  une liste de variables.
            * `signal_slider`:      la scrollbar correspondant aux
                                    différents signaux.
            * `previous_button`:    le bouton 'précédent'.
            * `next_button`:        le bouton 'suivant'.
            * `update_function`:    la fonction de mise à jour de 
                                    l'affichage.

            La mise à jour d el'affichage se fait par le callback 
            `update_function`. Dans sa version de base elle est exécutée
            par l'appel à la fonction `interactive`:

              out =  widgets.interactive(update_function,
                                        colname=variable_dropdown,
                                        sigpos=signal_slider)

            Si l'on modifie cette fonction il faudra d'abord stocker sa valeur
            puis appeler l'appeler avec les objets correspondants si on souhaite
            qu'ils restent actifs.

            :return: le dictionnaire d'éléments utiles à l'affichage décrit
            ci-dessus.
        """

        # Mise à jour du signal.
        nbmax = len(self.records)
        if nbmax==0:
            raise OpsetError(self.storename, "Opset is empty.")
        
        if (pos is not None) and (pos >= 0) and (pos < nbmax) and (pos != self.sigpos):
            self.sigpos = pos
        # On relit systématiquement le fichier au début au cas où une nouvelle colonne
        # serait ajoutée.
        self.df = pd.read_hdf(self.storename,self.records[self.sigpos])

        self.colname = get_colname(self.df.columns,name)
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
      
                           
        def update_plot(colname, sigpos):
            """ La fonction d'interactivité avec les gadgets."""
            
            self.colname = colname
            name, unit = nameunit(colname)
            
            # Pour éviter de relire deux fois le signal initial.
            if sigpos != self.sigpos:
                self.sigpos = sigpos
                self.df = pd.read_hdf(self.storename, self.records[sigpos])

            # Mise à jour des courbes.
            f.update_traces(selector=dict(name="value"),
                            x = self.df.index, y = self.df[self.colname])
            #scatter = f.data[0]
            #scatter.x = self.df.index
            #scatter.y = self.df[self.colname]
            if self.phase is not None:
                ind = self.df[self.phase]
                f.update_traces(selector=dict(name="phase"),
                                x = self.df.index[ind],
                                y = self.df[self.colname][ind])
                #scatter2 = f.data[1]
                #scatter2.x = self.df.index[ind]
                #scatter2.y = self.df[self.colname][ind]

            # Mise à jour des titres et labels.
            f.layout.title = name
            f.layout.xaxis.title = self.df.index.name + '[' + str(self.sigpos) + ']'
            f.layout.yaxis.title = unit
            f.layout.titlefont = {'color': "blue"}
            f.layout.xaxis.titlefont.color = "blue"
            f.layout.yaxis.titlefont.color = "blue"
         
        # Construction des gadgets interactifs.
        wd = widgets.Dropdown(options=self.df.columns,
                              value=self.colname,
                              description="Variable :")
        wbp = widgets.Button(description='Previous')
        wbn = widgets.Button(description='Next')
        ws = widgets.IntSlider(value=self.sigpos, min=0, max=nbmax-1, step=-1,
                               orientation='vertical',
                               description='Record',
                               layout=widgets.Layout(height='400px'))
        # Il suffira d'exécuter la commande suivante une fois que les gadgets
        # seront disposés à l'écran :
        #   out = widgets.interactive(update_plot, colname=wd, sigpos=ws)

        # Callbacks des boutons Previous et Next.
        def wb_on_click(b):
            if b.description == 'Previous':
                if ws.value > 0:
                    ws.value -= 1
            if b.description == 'Next':
                if ws.value < ws.max:
                    ws.value += 1

        wbp.on_click(wb_on_click)
        wbn.on_click(wb_on_click)
        
        # Pour la création de l'interface on peut encapsuler des boîtes.
        # boxes = widgets.VBox([widgets.HBox([wd, wbp, wbn]),
        #                       widgets.HBox([f, ws])])
        update_plot(self.colname, self.sigpos)
        return dict(variable_dropdown = wd,
                    signal_slider = ws,
                    previous_button = wbp,
                    next_button = wbn,
                    update_function = update_plot)
    
    
    def plot(self,phase=None,pos=None,name=None):
        """ Affichage de l'interface.
        
            La fonction plot commence par créer les différents éléments par
            un passage de ses paramètres à `make_figure`, puis elle doit
            mettre en oeuvre l'interactivité par un appel à `interactive`
            et construire le look de l'interface en positionnant les 
            objets. Il est aussi possible de modifier le `layout`de la 
            figure.

            En entrée, les mêmes paramètres que `make_figure`,
            et en sortie une organisation des éléments dans une boite.
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