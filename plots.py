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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    
        :param columns:  La liste des retours possibles (on peu passer un df).
        :param variable: Le début du nom rechercé dans la liste.
        :param default:  Un moyen de différentier si on veut une valeur 
                            ou rien quand rien n'est trouvé.
                            
        Par défaut la fonction renvoie la première donnée.
        En posant `Default=None` une entrée vide renvoie le vide.
    """
    
    if isinstance(columns,pd.DataFrame):
        df = columns
        columns = df.columns
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
    out = widgets.interactive_output(unit_plot, dict(unit=wu, variable=wv))
    boxes = widgets.VBox([widgets.HBox([wu,wv]),out])
    return boxes

    
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
    

def tsplot(df,cols=None,title=None):
    """ Affichage d'une série temporelle."""
    
    if isinstance(df,pd.Series):
        df = pd.DataFrame(df)
    fig = make_subplots(rows=1,cols=1)
    if not cols:
        cols = df.columns
    else:
        if isinstance(cols,str):
            cols = [cols]
        cols = [get_colname(df.columns,col) for col in cols]
        
    [fig.add_trace(go.Scatter(x=df.index,y=df[col],name=col),
                   row=1,col=1)
     for col in cols]
    fig.update_xaxes(
        title=df.index.name,
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="sec", step="second", stepmode="backward"),
                dict(count=1, label="min", step="minute", stepmode="backward"),
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))
    
    if len(cols)==1 or len(set(byunits(cols).keys()))==1:
        name,unit = nameunit(cols[0])
        fig.update_yaxes(title_text=unit, row=1, col=1)

    fig.update_layout(showlegend=True)
    if title:
        fig.update_layout(titlefont={'color': "blue"},
                          title_text=title)
    return fig


###########################################################################
# Affichages analytiques.

def pcacircle(df,pca=None,sample=0):
    """ Construit le cercle descriptif des composantes d'un ACP.
    
        L'analyse en composante principale peut être directement 
        passée en argument. Dans le cas contraire une analyse simple
        est faite.
    """
    
    X = StandardScaler().fit_transform(df.values)
    if not pca:
        pca = PCA().fit(X)

    cnames = ["PC{} ({:.1f}%)".\
              format(c,pca.explained_variance_ratio_[c-1]*100)
              for c in range(1,pca.n_components_+1)]
    Z = pca.transform(X)
    
    def update_circle(cname1,cname2):
        
        comp1 = cnames.index(cname1)
        comp2 = cnames.index(cname2)

        pc1 = pca.components_[comp1]
        pc2 = pca.components_[comp2]
        
        if sample>0:
            pts = np.random.choice(len(Z),int(len(Z)*sample),False)
            z1 = Z[pts,comp1]
            z2 = Z[pts,comp2]
        else:
            z1 = []
            z2 = []

        scalex = np.sqrt(pca.explained_variance_[comp1])
        scaley = np.sqrt(pca.explained_variance_[comp2])

        data = [go.Scatter(x = pc1*scalex, y = pc2*scaley, mode="markers+text",
                           text=df.columns, textposition="top right",
                           marker=dict(color="red", size=10), showlegend=False,
                           hoverinfo='skip')]
        data2 = [go.Scatter(x=[0,pc1[i]*scalex],y=[0,pc2[i]*scaley],mode="lines",
                            line=dict(color="red",width=1,dash="dot"),
                            name='var',
                            showlegend=False) for i in range(0,df.shape[1])]

        if sample>0:
            data3 = [go.Scatter(x = z1*scalex, y=z2*scaley, mode="markers",
                            marker=dict(color="black", opacity=0.15, size=5),
                            name='obs',
                            showlegend=False)]
        else:
            data3 = []
            
        shapes=[go.layout.Shape(type="circle",
                                xref="x",yref="y",
                                x0=-1,y0=-1,x1=1,y1=1,
                                line_color="LightBlue")]

        total_variance2 = pca.explained_variance_ratio_[comp1] + \
                          pca.explained_variance_ratio_[comp2]
        layout = go.Layout(title="Projection dans le plan PC{} x PC{} ({:.1f}%)".\
                          format(comp1+1,comp2+1,total_variance2*100),
                          xaxis_title=cnames[comp1],
                          yaxis_title=cnames[comp2],
                          shapes=shapes,
                          xaxis=dict(range=[-1.2,1.2]),
                          yaxis=dict(range=[-1.2,1.2],scaleanchor="x",scaleratio=1))
        fig = go.Figure(data+data2+data3,layout)
        fig.show()
        
    wx = widgets.Select(options=cnames,value=cnames[0],description='Abscisse')
    wy = widgets.Select(options=cnames,value=cnames[1],description='Ordonnée')
    
    out = widgets.interactive_output(update_circle,dict(cname1=wx,cname2=wy))
    boxes = widgets.VBox([widgets.HBox([wx,wy]),out])
    return boxes



