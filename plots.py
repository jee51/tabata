# -*- coding: utf-8 -*-
"""
PLOTS - Quelques affichages utiles.

La plupart de ces afficjhages utilisent des DataFrames pandas.

**Versions**

1.0.0 - Création et transfert des plots de bases de opset.py
"""

__date__ = "2020-05-09"
__version__ = '1.0.0'

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
        :param variable: une variable à afficher au lieu de la première
                            colonne de la table.
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


def byunitplot(df, yunit=None, title=""):
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
    
    def unit_plot(unit, variable):
        """ Fonction d'interactivité des gadgets."""
        if unit not in dnu:
            return
        if variable == 'All' or variable is None:
            cols = dnu[unit]
            data = [go.Scatter(x=df.index, y=df[col], name=nameunit(col)[0])
                    for col in cols]
        else:
            data = [go.Scatter(x=df.index, y=df[variable],
                               name=nameunit(variable)[0])]
        layout = go.Layout(title=title,
                           titlefont={'color': "blue"},
                           xaxis={'title': df.index.name,
                                  'titlefont': {'color': "blue"}},
                           yaxis={'title': unit,
                                  'titlefont': {'color': "blue"}},
                           showlegend=True)
        fig = go.Figure(data,layout)
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
 
    wu.observe(update_variables, 'value')
    widgets.interact(unit_plot, unit=wu, variable=wv)

    
def groupplot(df,title="",standardize=False):
    """ Un affichage superposant les courbes du DataFrame.
        
        :param df:          le DataFrame contenant les données.
        :param standardize: un booléen précisant s'il faut standardiser
                            les données.
        :param title:       un titre optionnel.
    """
    if standardize:
        data = [go.Scatter(x=df.index, 
                           y=(df[col]-df[col].mean())/df[col].std(),
                           name=col)
                    for col in df.columns 
                    if np.issubdtype(df[col].dtype,np.number) and df[col].std()>0]
    else:
        data = [go.Scatter(x=df.index, y=df[col], name=col)
                    for col in df.columns 
                    if np.issubdtype(df[col].dtype,np.number)]
    
    layout = go.Layout(title=title,
                       titlefont={'color': "blue"},
                       xaxis={'title': df.index.name,
                              'titlefont': {'color': "blue"}},
                       showlegend=True)
    fig = go.Figure(data,layout)
    fig.show()
    

def doubleplot(df1,df2=None,p=0.5,space=0.05,title=None):
    """ Affiche un plot en deux graphes liés.
        
        On passe soit deux DataFrames, soit un seul et les colonnes
        à afficher en haut dans `cols`.
    
        :param df1:      le DataFrame principal
        :param df2|cols: un autre DataFrame ou les colonnes à extraire du
                         premier.
        :param p:        la proportion de l'espace pour le premier graphe.
        :param space:    l'espacement vertical entre les deux praphes.
        :param title:    un titre optionnel.
        
        *Exemples*
        
            doubleplot(df,'ALT')
            doubleplot(df1,df2,0.3)
    """
    if isinstance(df2,str):
        df2 = [df2]
    if isinstance(df2,list):
        cols = [get_colname(df1.columns,c) for c in df2]
        df2 = df1.copy().drop(cols,axis=1)
        df1 = df1[cols]
    if isinstance(df1,pd.Series):
        df1 = pd.DataFrame(df1)
    if isinstance(df2,pd.Series):
        df2 = pd.DataFrame(df2)
    
    fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True)
    fig.update_yaxes(domain=(1-p, 1.0), row=1, col = 1)
    fig.update_yaxes(domain=(0.0, 1-p-space), row=2, col = 1)
    fig.update_xaxes(title_text=df1.index.name, 
                     titlefont={'color': "blue"}, row=2, col=1)
    
    [fig.add_trace(go.Scatter(x=df1.index,y=df1[col],name=col),
                   row=1,col=1)
     for col in df1.columns]
    if len(df1.columns)==1 or len(set(byunits(df2.columns).keys()))==1:
        name,unit = nameunit(df1.columns[0])
        fig.update_yaxes(title_text=unit, row=1, col=1)
    
    [fig.add_trace(go.Scatter(x=df2.index,y=df2[col],name=col),
                   row=2,col=1)
     for col in df2.columns]
    if len(df2.columns)==1 or len(set(byunits(df2.columns).keys()))==1:
        name,unit = nameunit(df2.columns[0])
        fig.update_yaxes(title_text=unit, row=2, col=1)
        
    if title:
        fig.update_layout(titlefont={'color': "blue"},
                          title_text=title)
    fig.update_layout(showlegend=True)
    fig.show()
    
###########################################################################


