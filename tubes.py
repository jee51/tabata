# -*- coding: utf-8 -*-
"""
TUBES - Tubes de cnfiance.

On crée un tube de confiance autour des données stockées dans
l'Opset. Le modèle du tube pourra alors servir comme fonction
de prédiction sur d'autres données.

**Versions**

1.0.1 - Création.


todo::
    - 
    
Created on Mon April 13 12:04:00 2020

@author: Jérôme Lacaille
"""

__date__ = "2020-04-13"
__version__ = '1.0.1'

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal
from sklearn import linear_model
from tabata.opset import Opset, nameunit, OpsetError

###########################################################################
#%% Fonctions auxiliaires.
def highlight(origin,extract,filename=None):
    """ Mise en évidence d'une extraction de données.
        
        Crée un Opset à partir de l'Opset original sur lequel on rajoute
        une `phase`égale aux instants de l'extraction.
        
        Si aucun nom de fihier n'est passé un nom temporaire est créé à
        partir de l'original suivi de '_E'.
    """

    if not isinstance(origin, Opset) or not isinstance(extract,Opset):
        raise OpsetError("Unknown","Need two Opsets as inputs")
    
    if len(origin) != len(extract):
        raise OpsetError(origin.storename,"Both lengths must be equal")
        
    if filename is None:
        i = origin.storename.rfind('.')
        filename = origin.storename[:i] + '_E' + origin.storename[i:]
    
    ds = Opset(filename).clean()
    for df in origin.iterator():
        dfe = extract[origin.sigpos]
        df["INTERVAL"] = np.isin(df.index,dfe.index)
        ds.put(df)
    ds.phase = "INTERVAL"
    
    return ds.rewind()
    
    
def plot(origin,pred):
    """ Superpose le tube sur les données d'origine.
    
        :param data: un Opset initial.
        :param tube: un Tube construit depuis l'origine.
    """
    
    if not isinstance(origin,Opset):
        raise OpsetError("Unknown","Should have an Opset as origin")
        
    if not isinstance(pred,Tube):
        raise OpsetError(origin.storename,"Prediction shold be a Tube")
        
    if len(origin) != len(pred):
        raise OpsetError(origin.storename,"Both inputs must have the same length")
        
    # ====================== Affichages ===========================                              
    f = make_subplots(rows=1, cols=1)
    f = go.FigureWidget(f)

    # Récupération de l'interface de l'Opset.
    e = origin.make_figure(f)
    old_update = e['update_function']
        
    # Affichage de la proba de présence
    pred.sigpos = origin.sigpos
    pred.colname = origin.colname
    z = pred.estimate()
    f.add_trace(go.Scatter(x=pred.df.index, y=z, opacity=0.7,
                           name='pred',
                           line=dict(color='darkgreen', 
                                     width=2)), 
                row=1, col=1)       
            
    # ---- Begin: Callback Interactive  ----
    def update_plot(colname, sigpos):
        """ Mise à jour de l'affichage."""
        old_update(colname,sigpos)
            
        pred.sigpos = origin.sigpos
        pred.colname = origin.colname
        z = pred.predict()
        f.update_traces(selector=dict(name='pred'),
                        x = self.df.index,
                        y = z)            
    # ---- End: Callback Interactive ----

    # On remplace la fonction d'update (que l'on avait d'abord copiée).
    e['update_function'] = update_plot 
    
    out = widgets.interactive(e['update_function'], 
                              colname=e['variable_dropdown'], 
                              sigpos=e['signal_slider'])
        
    boxes = widgets.VBox([widgets.HBox([e['variable_dropdown'], 
                                        e['previous_button'], 
                                        e['next_button']]),
                          widgets.HBox([f, e['signal_slider']])])
        
    return boxes
    
    
###########################################################################
#%% Un affichage interactif permettant de visualiser le contenu de la liste.
class Tube(Opset):
    """ L'Opset contenant les données à utiliser pour l'apprentissage.
    """
    
    def __init__(self, storename, phase=None, pos=0, name=""):
        """ Initialise les listes d'instants et d'opération."""
        Opset.__init__(self, storename, phase, pos, name)
        
        self._reg = None
        self.learn_params = dict(samples_percent=0.01) 
            
            
    def __repr__(self):
        """ Affiche le nombre de sélections."""

        return "{}\n" \
               "TUBE : ...".format(Opset.__repr__(self))
    
    def fit(self):
        """ Apprentissage d'un modèle."""
        
        if len(self) == 0:
            raise OpsetError(self.storename,"No data")
            
        colname = self.colname
        columns = self.df.columns
        cols = [c for c in columns if c != colname]
        
        p = self.learn_params['samples_percent']
        
        # Fabrication des données d'apprentissage.
        def find_best_parameters(p,colname,cols):
            X = []
            Y = []
            i0 = 0
            for df in self.iterator():
                n = len(df)
                pos = np.random.choice(np.arange(n),int(np.ceil(n*p)))
                df1 = df.iloc[pos]
                x = df1[cols]
                y = df1[colname]
                x.index = range(i0,i0+len(x))
                y.index = range(i0,i0+len(x))
                i0 = i0+len(x)
                Y.append(y)
                X.append(x)
            dfx = pd.concat(X)
            dfy = pd.concat(Y)

            # Création du modèle.
            reg = linear_model.LinearRegression()
            reg = reg.fit(dfx,dfy)

            return reg

        reg = find_best_parameters(p,colname,cols)
        self._reg = reg
        return reg
    
    
    def estimate(self):
        """ Renvoie la prédiction et le tube de confiance.
        
            :param filename: Si un nom de fichier est passé, alors la méthode
                             `load(filename)` est d'abord appelée.
                             
            :return: UN DataFrame contenant la variable prédite et ses bornes.
        """
        
        df = self.df

        colname = self.colname
        columns = self.df.columns
        cols = [c for c in columns if c != colname]
        
        x = df[cols]
        y = df[colname]
        z = self._reg.predict(x)
        
        return z
    
    # ====================== Affichages ===========================                              
    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Création de l'interface graphique.
        
            
        """

        # Récupération de l'interface de l'Opset.
        e = Opset.make_figure(self,f,phase,pos,name)
        
        # self.sigpos et self.colname sont mis à jour, 
        # ne pas utiliser ces variables ensuite.
        old_update = e['update_function']
        
        # Affichage de la proba de présence
        z = self.estimate()
        f.add_trace(go.Scatter(x=self.df.index, y=z, opacity=0.7,
                               name='pred',
                               line=dict(color='darkgreen', 
                                         width=2)), 
                    row=1, col=1)       
            
        # =================================================================
        # ---- Begin: Callback Interactive  ----
        def update_plot(colname, sigpos):
            """ Mise à jour de l'affichage.
            """
            old_update(colname,sigpos) # met à jour les positions.
            
            z = self.estimate()
            f.update_traces(selector=dict(name='pred'),
                            x = self.df.index,
                            y = z)            
        # ---- End: Callback Interactive ----

        # On remplace la fonction d'update (que l'on avait d'abord copiée).
        e['update_function'] = update_plot 
        return e

    
