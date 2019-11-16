# -*- coding: utf-8 -*-
"""
OPSET - Gestion des fichiers UDF5 comme des OPSETS de données.

Les affichages sont conditionnés pour s'exécuter dans un notebook Jupyter.

Created on Wed May  9 16:50:34 2018

@author: s068990
"""

import numpy as np
import pandas as pd

import ipywidgets as widgets
from IPython.display import display

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)


#%% Constantes.
SEP = '['  # Séparateur entre le nom et l'unité.


#%% Fonctions auxiliaires.
def nameunit(col,sep=SEP):
    """
    Renvoie le nom et l'unité d'une variable au format NOM[UNITE]
    """
    i = col.find(sep)
    if i == -1:
        return col,'-'
    return col[:i],col[i+1:-1]


def byunits(cols,sep=SEP):
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


#%% Affichages d'un signal d'un DataFrame.
def selplot(df):
    """
    SELPLOT - Affiche un signal parmis la liste des signaux disponibles.

    :param df: la table de données.
    """
    def selected_plot(col):
        name,unit = nameunit(col)
        data = [go.Scatter(x=df.index,y=df[col])]
        layout = go.Layout(title=name,
                           titlefont={'color':"blue"},
                           xaxis={'title':df.index.name,
                                  'titlefont':{'color':"blue"}},
                           yaxis={'title':unit,
                                  'titlefont':{'color':"blue"}})
        iplot(go.Figure(data=data,layout=layout),show_link=False)

    wd = widgets.Dropdown(options=df.columns, description="Variable :")
    widgets.interact(selected_plot,col=wd)


#%% Affichages d'un signal d'un DataFrame.
def byunitplot(df,title="",xunit="date"):
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
    
    def unit_plot(unit,variable='All'):
        if unit not in dnu:
            return
        if variable == 'All' or variable is None:
            cols = dnu[unit]
            data = [go.Scatter(x=df.index,y=df[col],name=nameunit(col)[0])
                    for col in cols]
        else:
            print(variable)
            data = [go.Scatter(x=df.index,y=df[variable],
                               name=nameunit(variable)[0])]
        layout = go.Layout(title=title,
                           titlefont={'color':"blue"},
                           xaxis={'title':xunit,
                                  'titlefont':{'color':"blue"}},
                           yaxis={'title':unit,
                                  'titlefont':{'color':"blue"}},
                           showlegend=True)
        iplot(go.Figure(data=data,layout=layout),show_link=False)
        
    def update_variables(*args):
        if wu.value == '?':
            wv.options = ['?']
            wv.value = '?'
            return
        wv.options = ['All']+dnu[wu.value]
        wv.value = 'All'
        
    wu = widgets.Dropdown(options=["?"]+units, description="Unité :")
    wv = widgets.Dropdown(options=['?'], 
                          description="Variables :")
    wu.layout = widgets.Layout(width='30%')
    wu.observe(update_variables,'value')
    b = widgets.HBox([wu,wv])
    out = widgets.interactive_output(unit_plot,dict(unit=wu,variable=wv))
    display(b,out)


#%% Parcours d'un OPSET HDF5
def dataiterator(storename, nbmax=None):
    """
    DATAITERATOR - Un itérateur sur un HDFStore pandas.
    Chaque itération renvoie la table suivante.

    :param storename: le chemin du fichier HDFStore.
    :param nbmax: le nombre maximum d'éléments pour l'itération.
    :return: (yield) un DataFrame pandas.
    """
    with pd.HDFStore(storename, mode='r') as store:
        records = store.keys()
    if nbmax is not None and len(records) > nbmax:
        records = records[:nbmax]

    return (pd.read_hdf(storename,rec) for rec in records)


#%% Affichage des distributions de données.
def repartition_plot(dfiterator, width=5):
    """
    REPARTITION_PLOT - Affiche la distribution des données sur une échelle
    de temps.

    Des barres représentent les instants d'acquisition.
    La hauteur de chaque barre correspond à la fréquence.
    Les barres sont élargies artificiellement d'une proportion
    correspondant au (width%)-ième de la largeur du graphe mutiplié par le
    nombre de mesures non nulles acquises.
    Les barres sont transparentes et se superposent ce qui permet de mieux
    se rendre compte du volume acqui.

    :param dfiterator: un itérateur sur une liste de DataFrames.
    :param width: une proportion (en %) de la durée totale d'acquisition
    qui est utilisée pour élargir les barres en fonction du nombre de
    variables collectées.
    """

    # Création de la table descriptive des données.
    data = dict(name=[], freq=[], start=[], stop=[],
                nbvar=[], nbnan=[], nbcte=[], nbpts=[])
    for df in dfiterator:
        data['name'].append(df.index.name)
        t0 = df.index[0]
        t1 = df.index[-1]
        n = len(df.index)
        if n == 1 or t0 == t1:
            f = 0
        else:
            f = (n - 1) / (t1 - t0).total_seconds()
        data['freq'].append(f)
        data['start'].append(t0)
        data['stop'].append(t1)
        data['nbvar'].append(len(df.columns))
        # Récupération et test du contenu.
        df = df.replace([-999.0, np.inf, -np.inf], np.nan)
        data['nbnan'].append(df.isna().all(axis=0).sum())
        data['nbcte'].append(np.sum(df.max(axis=0) == df.min(axis=0)))
        data['nbpts'].append(len(df))

    tb = pd.DataFrame(data)

    # Affichage.
    m0 = tb['start'].min()  # Plus petite valeur d'enregistrement.
    m1 = tb['stop'].max()   # Plus grande valeur d'enregistrement.

    dt = width*(m1 - m0)/100  # Largeur maximale des bandes.
    M = tb['nbvar'].max()   # Nombre maximal de variables dans un
                            # enregistrement.
    data = [go.Scatter(x=[m0-dt, m1+dt], y=[0, 0],
                       mode='text', text=[''])]  # Dummy.
    rectangles = []
    for k in range(len(tb)):
        f = tb.loc[k, 'freq']
        nbcol = tb.loc[k, 'nbvar'] - tb.loc[k, 'nbcte'] - tb.loc[k, 'nbnan']
        t0 = tb.loc[k, 'start'] - dt*(0.1 + nbcol/M)
        t1 = tb.loc[k, 'stop'] + dt*(0.1 + nbcol/M)
        r = dict(type='rect',
                 x0=t0, x1=t1, y0=0, y1=f,
                 line=dict(color='rgba(128,0,0,0.1)', width=1),
                 fillcolor='rgba(128,0,0,0.1)')
        rectangles.append(r)
    layout = go.Layout(title="Répartition des données",
                       xaxis={"title": "Date"},  # ,'rangeslider':dict()},
                       yaxis=dict(title="Fréquences [Hz]"),
                       shapes=rectangles,
                       hovermode=False,
                       showlegend=False)
    iplot(go.Figure(data=data, layout=layout), show_link=False)

#%% Affichage d'opset et selection de points.
class Plotter:
    """
    Plotter - Un affichage interactif utilisant plotly qui permet de
    sélectionner des pointrs dans un OPSET.
    """

    def __init__(self,storename):
        """
        Sauvegarde du datastore utilisé et création du dictionnaire des
        éléments à conserver.

        :param storename: le datatore correspondant à l'OPSET.
        """

        self.storename = storename
        self.records = list()
        self.sigpos = 1
        self.colname = ""
        self.sel_instants = dict()
        self.op_viewed = set()

    def __repr__(self):
        """
        Affichage du nom de l'OPSET et de la liste des instants selectionnés.
        :return: la description du sélecteur.
        """

        ops = len(self.op_viewed)
        pts = len(self.sel_instants)
        return "({}) {} instant(s) sélectionnés parmis {} courbes " \
               "observées".format(self.storename, pts, ops)

    def renew(self):
        """
        RENEW - Réinitialise la liste des sélections.
        """

        self.op_viewed.clear()
        self.sel_instants.clear()

    def plot(self):
        """
        PLOT - Lance l'affichage interactif des courbes et accepte la
        selection.

        :return: les boites à afficher par jupyter.
        """

        with pd.HDFStore(self.storename, mode='r') as store:
            records = store.keys()
            df = store[records[0]]
            nbmax = len(records)
            col = df.columns[0]

        self.records = records
        self.colname = col
        self.sigpos = 1
        name, unit = nameunit(self.colname)
        df = pd.read_hdf(self.storename, records[self.sigpos - 1])

        data = [go.Scatter(x=df.index, y=df[self.colname])]
        layout = go.Layout(width=500, height=400,
                           title=name,
                           titlefont={'color': "blue"},
                           xaxis={'title': df.index.name + '[' + str(
                               self.sigpos) + ']',
                                  'titlefont': {'color': "blue"}},
                           yaxis={'title': unit,
                                  'titlefont': {'color': "blue"}})
        f = go.FigureWidget(data, layout)

        def update_plot(colname,sigpos):
            self.colname = colname
            self.sigpos = sigpos
            if len(self.sel_instants)>0:
                self.op_viewed.add(sigpos)
            name, unit = nameunit(colname)
            df = pd.read_hdf(self.storename, records[sigpos - 1])

            f.layout.shapes = []
            scatter = f.data[0]
            scatter.x = df.index
            scatter.y = df[colname]
            if sigpos in self.sel_instants:
                i = self.sel_instants[sigpos][1]
                x0 = df.index[i]
                y0 = min(df[colname])
                y1 = max(df[colname])
                shapes = [{'type':'line',
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


        wd = widgets.Dropdown(options=df.columns, description="Variable :")
        wbp = widgets.Button(description='Previous')
        wbn = widgets.Button(description='Next')
        ws = widgets.IntSlider(value=1, min=1, max=nbmax, step=-1,
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
            self.op_viewed.add(self.sigpos)
            i0 = points.point_inds[0]
            x0 = trace.x[i0]
            y0 = min(trace.y)
            y1 = max(trace.y)
            trace.parent.layout.shapes = [{'type':'line',
                                'x0': x0,
                                'y0': y0,
                                'x1': x0,
                                'y1': y1,
                                'line': {'color': 'rgb(171, 50, 96)',
                                         'width': 2,
                                         'dash': 'dashdot'}}]
            # print("colname=",self.colname, "  sigpos=",self.sigpos,
            #      "  point_inds=", i0)
            self.sel_instants[self.sigpos] = (self.colname, i0)

        scatter = f.data[0]
        scatter.on_click(selection_fn)

        boxes = widgets.VBox([widgets.HBox([wd, wbp, wbn]), widgets.HBox([
            f, ws])])
        return boxes

#%% Exemple
# Ce code est à exécuter dans un notebook Jupyter par
# %run ~/wrk/jl/opset.py
if __name__ == "__main__":
    from plotly.offline import init_notebook_mode
    init_notebook_mode(connected=True)

    df = pd.DataFrame(np.random.randn(10,5),
                       columns=['A[x]','B[x]','C[y]','D[x]','E[y]'])

    # print('Affichage courbe par courbe.')
    # selplot(df)
    #
    # print('Affichage par unité.')
    # byunitplot(df)

