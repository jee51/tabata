# -*- coding: utf-8 -*-
"""
OPSET - Gestion des liste d'observations stockées dans des fichiers au format HDF5.

Chaque observation est un dataFrame panas. Elle est nommée, le nom de l'enregistrement
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

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go

#%% Fonctions auxiliaires.
def nameunit(col,sep="["):
    """
    Renvoie le nom et l'unité d'une variable au format NOM[UNITE]
    """
    i = col.find(sep)
    if i == -1:
        return col,'-'
    return col[:i],col[i+1:-1]

def byunits(cols,sep="["):
    """
    Récupère un dictionnaire dont les clés sont les unités uniques et
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


# %% Affichages d'un signal d'un DataFrame.
def selplot(df, variable=None):
    """
    SELPLOT - Affiche un signal parmis la liste des signaux disponibles.

    :param df: la table de données.
    """

    def selected_plot(col):
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

    if variable is not None:
        columns = list(df.columns)
        subs = [r for r in columns if variable in r]
        if len(subs)>0:
            variable = subs[0]
        else:
            variable = columns[0]
    else:
        variable = df.columns[0]
    if (variable is None) or (variable not in df.columns):
        variable = df.columns[0]
    wd = widgets.Dropdown(options=df.columns, value=variable, description="Variable :")
    widgets.interact(selected_plot, col=wd)


# %% Affichages d'un signal d'un DataFrame.
def byunitplot(df, yunit=None, xunit="date", title=""):
    """
    BYUNITPLOT - Affiche les signaux en fonction de leur unité.

    Au début l'affichage est vide et une question est posée '?',
    en choisissant une unité on affiche toutes les courbes correspondant.
    On peut aussi sélectionner quelques courbes en cliquant sur les légendes.
    Un racourci permet de n'afficher qu'une seule courbe.

    :param df: la table de données.
    :param title: un titre à la figure.
    :param xunit: l'unité de date.
    """
    dnu = byunits(df)
    units = list(dnu.keys())

    def unit_plot(unit, variable='All'):
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
        wv.options = ['All'] + dnu[wu.value]
        wv.value = 'All'

    if (yunit is None) or (yunit not in units):
        yunit = units[0]
    wu = widgets.Dropdown(options=units, value=yunit, description="Unité :")
    wv = widgets.Dropdown(options=['All']+dnu[units[0]], value='All', description="Variables :")
    #wu = widgets.Dropdown(options=["?"] + units, description="Unité :")
    #wv = widgets.Dropdown(options=['?'], description="Variables :")
    wu.layout = widgets.Layout(width='30%')
    wu.observe(update_variables, 'value')
    b = widgets.HBox([wu, wv])
    out = widgets.interactive_output(unit_plot, dict(unit=wu, variable=wv))
    return widgets.VBox([b,out])


#%% Un affichage interactif permettant de visualiser le contenu de la liste.
class Selector:
    """
    SELECTOR - Un affichage interactif utilisant plotly qui permet de
    sélectionner des points dans une liste d'observations.
    """

    def __init__(self, storename, sigpos=1, colname=""):
        """
        Sauvegarde du datastore utilisé et création du dictionnaire des
        éléments à conserver.

        :param storename: le datatore correspondant à l'OPSET.
        """
        
        if not os.path.isfile(storename): # Il faut créer le fichier.
            newstore = pd.HDFStore(storename,mode='w')
            newstore.close()
            
        self.storename = storename
        with pd.HDFStore(self.storename, mode='r') as store:
            self.records = store.keys()
            nbmax = len(self.records)
            if (sigpos < 1) or (sigpos > nbmax):
                sigpos = 1
            if nbmax>0:
                self.df = store[self.records[sigpos-1]]
                if colname not in self.df.columns:
                    colname = self.df.columns[0]
            else:
                self.df = None
                colname = ""
                
        self.sigpos = sigpos
        self.colname = colname
        self.sel_instants = dict()
        self.op_viewed = set()
        self.phase = ""

    def __repr__(self):
        """
        Affichage du nom de l'OPSET et de la liste des instants selectionnés.
        :return: la description du sélecteur.
        """

        ops = len(self.op_viewed)
        pts = len(self.sel_instants)
        return "({}) {} instant(s) sélectionnés parmis {} courbes " \
               "observées".format(self.storename, pts, ops)

    def clear_selection(self):
        """
        RENEW - Réinitialise la liste des sélections.
        """

        self.op_viewed.clear()
        self.sel_instants.clear()

    def iterator(self, first=1,last=None):
        """
        Itération sur les éléments du HDF5SET.
        :param first=1: le premier élément  itérer (commence à 1)
        :param last=None: le dernier élément.
        :return: le DataFrame
        """
        if last is None:
            last = len(self.records)
        if first<1 or first>last:
            first=1

        for i in range(first-1,last):
            self.sigpos = i+1
            rec = self.records[i]
            self.df = pd.read_hdf(self.storename, rec)
            yield self.df
    
    def current_record(self):
        """
        CURRENT-RECORD - Renvoie le nom de l'enregistrement courant.
        """
        if len(self.records)==0:
            return ""
        else:
            return self.records[self.sigpos-1]
        
    def put(self,df,record=None):
        """
        PUT - Stocke le signal dans le fichier.
        
        :param df: le signal à stocker.
        :param record: l'enregistrement du signal 
        
        Si aucun nom d'enregistrement n'est donné on regarde df.index.name.
        Si le nom de l'enregistrement n'existe pas il est rajouté.
        """
        if record is None:
            if (not df.index.name) or len(df.index.name)==0:
                raise ValueError("Record name is missing.")
            record = df.index.name

        if record in self.records:
            self.sigpos = self.records.index(record)+1
        else:
            self.sigpos = len(self.records)+1
            self.records.append(record)

        if (not df.index.name) or len(df.index.name)==0:
            df.index.name = record
        
        self.df = df
        df.to_hdf(self.storename,record)

    def plot(self,phase=None,sigpos=None,colname=None):
        """
        PLOT - Lance l'affichage interactif des courbes et accepte la
        selection.
        
        :return: les boites à afficher par jupyter.
        """
        
        nbmax = len(self.records)
        if nbmax==0:
            return
        
        if (sigpos is not None) and (sigpos >= 1) and (sigpos <= nbmax):
            self.sigpos = sigpos
        # On relit systématiquement le fichier au début au cas où une nouvelle colonne
        # serait ajoutée.
        self.df = pd.read_hdf(self.storename,self.records[self.sigpos-1])
        df = self.df
        if colname in df.columns:
            self.colname = colname
        elif self.colname not in df.columns:
            self.colname = df.columns[0]
        if (phase is not None) and (phase not in df.columns):
            phase = None

        name, unit = nameunit(self.colname)

        data = [go.Scatter(x=df.index, y=df[self.colname])]
        if phase is not None:
            ind = df[phase]
            data.append(go.Scatter(x=df.index[ind], y=df[self.colname][ind],
                                   line={'color':'red'}))
        layout = go.Layout(width=500, height=400,
                           title=name,
                           titlefont={'color': "blue"},
                           showlegend=False,
                           xaxis={'title': df.index.name + '[' + str(
                               self.sigpos) + ']',
                                  'titlefont': {'color': "blue"}},
                           yaxis={'title': unit,
                                  'titlefont': {'color': "blue"}})
        f = go.FigureWidget(data, layout)

        def update_plot(colname, sigpos):
            self.colname = colname
            self.sigpos = sigpos
            if len(self.sel_instants) > 0:
                self.op_viewed.add(sigpos)
            name, unit = nameunit(colname)
            self.df = pd.read_hdf(self.storename, self.records[sigpos - 1])
            df = self.df

            f.layout.shapes = []
            scatter = f.data[0]
            scatter.x = df.index
            scatter.y = df[colname]
            if phase is not None:
                ind = df[phase]
                scatter2 = f.data[1]
                scatter2.x = df.index[ind]
                scatter2.y = df[colname][ind]
            if sigpos in self.sel_instants:
                i = self.sel_instants[sigpos][1]
                x0 = df.index[i]
                y0 = min(df[colname])
                y1 = max(df[colname])
                shapes = [{'type': 'line',
                           'x0': x0,
                           'y0': y0,
                           'x1': x0,
                           'y1': y1,
                           'line': {'color': 'rgb(171, 50, 96)',
                                    'width': 2,
                                    'dash': 'dashdot'}}]
            else:
                shapes = []
            f.layout.shapes = shapes

            f.layout.title = name
            f.layout.xaxis.title = df.index.name + '[' + str(sigpos) + ']'
            f.layout.yaxis.title = unit
            f.layout.titlefont = {'color': "blue"}
            f.layout.xaxis.titlefont.color = "blue"
            f.layout.yaxis.titlefont.color = "blue"

        wd = widgets.Dropdown(options=df.columns,
                              value=self.colname,
                              description="Variable :")
        wbp = widgets.Button(description='Previous')
        wbn = widgets.Button(description='Next')
        ws = widgets.IntSlider(value=self.sigpos, min=1, max=nbmax, step=-1,
                               orientation='vertical',
                               description='Record',
                               layout=widgets.Layout(height='400px'))
        out = widgets.interactive(update_plot, colname=wd, sigpos=ws)

        def wb_on_click(b):
            if b.description == 'Previous':
                if ws.value > 1:
                    ws.value -= 1
            if b.description == 'Next':
                if ws.value < ws.max:
                    ws.value += 1

        wbp.on_click(wb_on_click)
        wbn.on_click(wb_on_click)

        def selection_fn(trace, points, selector):
            if len(points.point_inds)>0:
                self.op_viewed.add(self.sigpos)
                i0 = points.point_inds[0]
                x0 = trace.x[i0]
                scatter1 = trace.parent.data[0]
                x1 = scatter1.x
                i1 = int(min(np.argwhere(x1>=x0)))
                y0 = min(scatter1.y)
                y1 = max(scatter1.y)
                trace.parent.layout.shapes = [{'type': 'line',
                                               'x0': x0,
                                               'y0': y0,
                                               'x1': x0,
                                               'y1': y1,
                                               'line': {'color': 'rgb(171, 50, 96)',
                                                        'width': 2,
                                                        'dash': 'dashdot'}}]
                # print("colname=",self.colname, "  sigpos=",self.sigpos,
                #      "  point_inds=", i0)
                self.sel_instants[self.sigpos] = (self.colname, i0, i1)
        
        scatter = f.data[0]
        scatter.on_click(selection_fn)
        if phase is not None:
            scatter2 = f.data[1]
            scatter2.on_click(selection_fn)

        boxes = widgets.VBox([widgets.HBox([wd, wbp, wbn]),
                              widgets.HBox([f, ws])])
        return boxes
        
