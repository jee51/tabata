# -*- coding: utf-8 -*-
"""
PLOTS - Quelques affichages utiles.

La plupart de ces afficjhages utilisent des DataFrames pandas.

**Versions**

1.0.0 - Création et transfert des plots de bases de opset.py
"""

__date__ = "2020-05-09"
__version__ = '1.1.1

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

"""
if pio.renderers.default == "vscode":
    # Ca ne marche pas avec vscode !
    init_notebook_mode(connected=True)
    pio.renderers.default = "notebook"
else:
    init_notebook_mode(connected=True) # Comportement par défaut.
"""

# Trouvé sur stackoverflow (https://stackoverflow.com/questions/64849484/display-plotly-plot-inside-vs-code)
#pio.renderers.default = "notebook"
#pio.renderers.default = "vscode"
#init_notebook_mode(connected=True)

###########################################################################
# Fonctions auxiliaires.
###########################################################################
def nameunit(col,sep='['):
    """ Renvoie le nom et l'unité d'une variable au format NOM[UNITE]"""
    i = col.find(sep)
    if i == -1:
        return col,'-'
    return col[:i],col[i+1:-1]


def byunits(cols,sep='['):
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
# Fonctions d'affichage de signaux.
###########################################################################
def selplotc(df, variable=None, sep='['):
    """ Affiche un signal parmis la liste des signaux disponibles.

        :param df:       la table de données.
        :param variable: une variable à afficher au lieu de la première
                            colonne de la table.
    """

    def selected_plot(col):
        """ La fonction d'interactivité de `selplot`.
        
            C'est cette fonction qui définit notamment le style du titre et des axes.
        """
        name, unit = nameunit(col,sep)
        data = [go.Scatter(x=df.index, y=df[col])]
        layout = go.Layout(title={'text': name, 'font': {'color': "blue"}},
                   xaxis={'title': {'text': df.index.name, 'font': {'color': "blue"}}},
                   yaxis={'title': {'text': unit, 'font': {'color': "blue"}}})
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    variable = get_colname(list(df.columns),variable)
    wd = widgets.Dropdown(options=df.columns, value=variable, description="Variable :")
    out = widgets.interactive(selected_plot, col=wd)
    return out

def selplot(df, variable=None, sep='['):
    """ Affiche un signal parmis la liste des signaux disponibles.

        :param df:       la table de données.
        :param variable: une variable à afficher au lieu de la première
                            colonne de la table.
    """

    f = make_subplots(rows=1, cols=1)
    f = go.FigureWidget(f)
    variable = get_colname(list(df.columns),variable)
    name, unit = nameunit(variable,sep)

    f.add_trace(go.Scatter(x=df.index, y=df[variable],name="value"),
                    row=1,col=1)
    f.update_layout(title={'text': name, 'font': {'color': "blue"}},
                    xaxis={'title': {'text': df.index.name, 'font': {'color': "blue"}}},
                    yaxis={'title': {'text': unit, 'font': {'color': "blue"}}})
                   
    def selected_plot(col):
        """ La fonction d'interactivité de `selplot`.
        
            C'est cette fonction qui définit notamment le style du titre et des axes.
        """
        name, unit = nameunit(col,sep)
        f.update_traces(selector=dict(name="value"),
                            x = df.index, y = df[col])
        f.update_layout(title={'text': name, 'font': {'color': "blue"}},
                        xaxis={'title': {'text': df.index.name, 'font': {'color': "blue"}}},
                        yaxis={'title': {'text': unit, 'font': {'color': "blue"}}})
        f.show()
        
    wd = widgets.Dropdown(options=df.columns, value=variable, description="Variable :")
    out = widgets.interactive(selected_plot, col=wd)
    boxes = widgets.VBox([out,f])
    return boxes
    

def selplotm(df, variable=None, sep='['):
    """ Affiche un signal parmis la liste des signaux disponibles.
        Cet affichage utilise Matplotlib

        :param df:       la table de données.
        :param variable: une variable à afficher au lieu de la première
                            colonne de la table.
    """

    def selected_plot(col):
        """ La fonction d'interactivité de `selplot`.
        
            C'est cette fonction qui définit notamment le style du titre et des axes.
        """
        name, unit = nameunit(col,sep)
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(9,4))
        sns.lineplot(data=df,x=df.index,y=col)
        plt.ylabel(unit)
        plt.title(name)

    variable = get_colname(list(df.columns),variable)
    wd = widgets.Dropdown(options=df.columns, value=variable, description="Variable :")
    out = widgets.interactive(selected_plot, col=wd)
    return out

###########################################################################
# Fonctions d'affichage de signaux par unité
###########################################################################
def byunitplot(df, yunit=None, title="", sep='['):
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

    f = make_subplots(rows=1, cols=1)
    f = go.FigureWidget(f)
    dnu = byunits(df,sep)
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
        f.data = []
        [f.add_trace(trace) for trace in data]
        f.update_layout(layout)

    unit_plot("m/s","All")

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
    boxes = widgets.VBox([widgets.HBox([wu,wv]),f])
    return boxes


def byunitplotm(df, yunit=None, title="", sep='['):
    """
    Affiche les signaux du DataFrame ayant la même unité, avec matplotlib.

    :param df: DataFrame contenant les signaux
    :param yunit: unité cible à afficher (ex: 'm', 'K', etc.)
    :param title: titre de la figure
    :param sep: séparateur pour parser les noms/units (par défaut '[')
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    dnu = byunits(df.columns, sep)
    units = list(dnu.keys())

    if yunit is None or yunit not in dnu:
        raise ValueError(f"Unité '{yunit}' non trouvée. Disponibles : {units}")

    cols = dnu[yunit]

    plt.figure(figsize=(12, 6))
    for col in cols:
        name, _ = nameunit(col, sep)
        plt.plot(df.index, df[col], label=name)

    plt.title(title or f"Signaux avec unité [{yunit}]", color="blue")
    plt.xlabel(df.index.name or "Index", color="blue")
    plt.ylabel(f"[{yunit}]", color="blue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
###########################################################################
# Fonctions d'affichage de signaux par groupes 
###########################################################################
def groupplot(df,title="",standardize=False):
    """ Un affichage superposant les courbes du DataFrame.
        
        :param df:          le DataFrame contenant les données.
        :param standardize: un booléen précisant s'il faut standardiser
                            les données.
        :param title:       un titre optionnel.
    """
    # JL: On peut factoriser la standardisation.
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


def groupplotm(df, title="", standardize=False):
    """
    Un affichage superposant les courbes du DataFrame.

    :param df: DataFrame avec les signaux
    :param title: titre de la figure
    :param standardize: si True, standardise chaque série (centrée réduite)
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    plt.figure(figsize=(12, 6))

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            serie = df[col].dropna()
            if standardize and serie.std() > 0:
                values = (serie - serie.mean()) / serie.std()
            else:
                values = serie
            plt.plot(df.index, values, label=col)

    plt.title(title or "Courbes superposées", color="blue")
    plt.xlabel(df.index.name or "Index", color="blue")
    plt.ylabel("Valeurs standardisées" if standardize else "Valeurs", color="blue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
###########################################################################
# Fonctions d'affichage de signaux doubles.
###########################################################################
def doubleplot(df1,df2=None,p=0.5,space=0.05,title=None):
    """ Affiche un plot en deux graphes liés.
        
        On passe soit deux DataFrames, soit un seul et les colonnes
        à afficher en haut dans `cols`.
    
        :param df1:      le DataFrame principal
        :param df2|cols: un autre DataFrame ou les colonnes à extraire du
                         premier.
        :param p:        la proportion de l'espace pour le premier graphe.
        :param space:    l'espacement vertical entre les deux graphes.
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
    # JL: Il faudrait standardiser si les unités sont différentes.
    if len(df1.columns)==1 or len(set(byunits(df2.columns).keys()))==1:
        name,unit = nameunit(df1.columns[0])
        fig.update_yaxes(title_text=unit, row=1, col=1)
    
    [fig.add_trace(go.Scatter(x=df2.index,y=df2[col],name=col),
                   row=2,col=1)
     for col in df2.columns]
    # JL: Il faudrait standardiser si les unités sont différentes.
    if len(df2.columns)==1 or len(set(byunits(df2.columns).keys()))==1:
        name,unit = nameunit(df2.columns[0])
        fig.update_yaxes(title_text=unit, row=2, col=1)
        
    if title:
        fig.update_layout(titlefont={'color': "blue"},
                          title_text=title)
    fig.update_layout(showlegend=True)
    fig.show()


def doubleplotm(df1, df2=None, p=0.5, space=0.05, title=None, sep='['):
    """
    Affiche deux sous-graphiques verticaux liés par l'axe X (matplotlib).

    :param df1: DataFrame principal ou série
    :param df2: Autre DataFrame ou liste de colonnes à extraire de df1
    :param p: Proportion de hauteur pour le haut
    :param space: Espace entre les deux sous-graphiques
    :param title: Titre de la figure
    :param sep: Séparateur pour nom + unité
    """
    # Traitement des formats d'entrée
    if isinstance(df2, str):
        df2 = [df2]
    if isinstance(df2, list):
        cols = [get_colname(df1.columns, c) for c in df2]
        df2 = df1.copy().drop(cols, axis=1)
        df1 = df1[cols]
    if isinstance(df1, pd.Series):
        df1 = pd.DataFrame(df1)
    if isinstance(df2, pd.Series):
        df2 = pd.DataFrame(df2)

    # Préparation de la figure avec GridSpec
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(100, 1)
    h1 = int(p * 100)
    h2 = 100 - h1 - int(space * 100)

    ax1 = fig.add_subplot(gs[:h1, 0])
    ax2 = fig.add_subplot(gs[-h2:, 0], sharex=ax1)

    # Courbes du haut
    for col in df1.columns:
        ax1.plot(df1.index, df1[col], label=col)
    if len(df1.columns) == 1 or len(set(byunits(df1.columns, sep).keys())) == 1:
        name, unit = nameunit(df1.columns[0], sep)
        ax1.set_ylabel(unit)
    ax1.set_title(title or "")
    ax1.legend()
    ax1.grid(True)

    # Courbes du bas
    for col in df2.columns:
        ax2.plot(df2.index, df2[col], label=col)
    if len(df2.columns) == 1 or len(set(byunits(df2.columns, sep).keys())) == 1:
        name, unit = nameunit(df2.columns[0], sep)
        ax2.set_ylabel(unit)
    ax2.set_xlabel(df1.index.name or "Index")
    ax2.legend()
    ax2.grid(True)

    # plt.tight_layout()
    # pour éviter un warning tight layout :
    plt.subplots_adjust(hspace=0.25)  # marge verticale
    plt.show()
    
###########################################################################
# Fonctions d'affichage de séries temporelles.
###########################################################################
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

def tsplotm(df, cols=None, title=None, sep='['):
    """
    Affiche une ou plusieurs séries temporelles avec matplotlib.

    :param df: DataFrame contenant les données
    :param cols: nom(s) de colonnes à afficher (ou None pour toutes)
    :param title: titre de la figure
    :param sep: séparateur pour détecter les unités (par défaut '[')
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    # Sélection des colonnes
    if cols is None:
        cols = df.columns
    elif isinstance(cols, str):
        cols = [cols]
    cols = [get_colname(df.columns, c) for c in cols]

    # Début du tracé
    plt.figure(figsize=(12, 6))
    for col in cols:
        plt.plot(df.index, df[col], label=col)

    # Déduction d'une unité unique si possible
    units = list(byunits(cols, sep).keys())
    y_label = f"[{units[0]}]" if len(units) == 1 else "Valeurs"

    plt.title(title or "Série temporelle", color="blue")
    plt.xlabel(df.index.name or "Index", color="blue")
    plt.ylabel(y_label, color="blue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

###########################################################################
# Affichages d'une PCA.
def pcacircle(df,pca=None,sample=0):
    """ Construit le cercle descriptif des composantes d'un ACP.
    
        L'analyse en composante principale peut être directement 
        passée en argument. Dans le cas contraire une analyse simple
        est faite.
    """
    
    # JL : Qu'en est-il des variables non numériques ?
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

def pcacirclem(df, pca=None, comp1=1, comp2=2, sample=0, sep='['):
    """
    Affiche le cercle des corrélations d'une ACP avec matplotlib.

    :param df: DataFrame de données numériques
    :param pca: instance PCA déjà calculée (sinon elle est ajustée automatiquement)
    :param comp1: numéro de la composante horizontale (commence à 1)
    :param comp2: numéro de la composante verticale (commence à 1)
    :param sample: fraction des observations à projeter (0 = aucune)
    :param sep: séparateur de noms/units pour affichage
    """
    X = StandardScaler().fit_transform(df.values)
    if pca is None:
        pca = PCA().fit(X)

    Z = pca.transform(X)
    comp1 -= 1
    comp2 -= 1

    scalex = np.sqrt(pca.explained_variance_[comp1])
    scaley = np.sqrt(pca.explained_variance_[comp2])

    pc1 = pca.components_[comp1]
    pc2 = pca.components_[comp2]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cercle de corrélation
    circle = plt.Circle((0, 0), 1, color='lightblue', fill=False, linestyle='--')
    ax.add_patch(circle)

    # Vecteurs des variables
    for i, var in enumerate(df.columns):
        x = pc1[i] * scalex
        y = pc2[i] * scaley
        ax.arrow(0, 0, x, y, color='red', alpha=0.7, head_width=0.02, length_includes_head=True)
        ax.text(x * 1.05, y * 1.05, var, color='red', fontsize=10)

    # Observations (optionnel)
    if sample > 0:
        n = int(len(Z) * sample)
        pts = np.random.choice(len(Z), n, replace=False)
        ax.scatter(Z[pts, comp1] / scalex, Z[pts, comp2] / scaley,
                   color='black', alpha=0.15, s=10, label="Observations")

    # Mise en forme
    total_var = 100 * (pca.explained_variance_ratio_[comp1] +
                       pca.explained_variance_ratio_[comp2])
    ax.set_title(f"Plan PC{comp1+1} x PC{comp2+1} ({total_var:.1f} % de variance)", color='blue')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f"PC{comp1+1} ({pca.explained_variance_ratio_[comp1]*100:.1f}%)")
    ax.set_ylabel(f"PC{comp2+1} ({pca.explained_variance_ratio_[comp2]*100:.1f}%)")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)

    plt.tight_layout()
    plt.show()

