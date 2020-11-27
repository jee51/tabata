# -*- coding: utf-8 -*-
"""
TUBES - Tubes de cnfiance.

On crée un tube de confiance autour des données stockées dans
l'Opset. Le modèle du tube pourra alors servir comme fonction
de prédiction sur d'autres données.

**Versions**

1.0.1 - Création.
1.0.2 - Tabs pour l'apprentissage.
1.0.3 - Utilisation de super()


todo::
    - 
    
Created on Mon April 13 12:04:00 2020

@author: Jérôme Lacaille
"""

__date__ = "2020-05-10"
__version__ = '1.0.3'

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal
from sklearn import linear_model

from .opset import Opset, OpsetError
from.plots import nameunit, get_colname

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
    sigpos = origin.sigpos
    for df in origin:
        dfe = extract[origin.sigpos]
        df["INTERVAL"] = np.isin(df.index,dfe.index)
        ds.put(df)
    origin.rewind(sigpos)
    ds.phase = "INTERVAL"
    
    return ds.rewind()
    
    
###########################################################################
#%% Un affichage interactif permettant de visualiser le contenu de la liste.
class AppTube(Opset):
    """ L'Opset contenant les données à utiliser pour l'apprentissage.
    """
    
    def __init__(self, origin, pred):
        """ Initialise les listes d'instants et d'opération."""
        super().__init__(origin.storename, 
                         origin.phase, origin.sigpos, origin.colname)
        
        self.pred = pred            
            
    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Création de l'interface graphique."""

        # Récupération de l'interface de l'Opset.
        e = Opset.make_figure(self,f,phase,pos,name)
        
        # self.sigpos et self.colname sont mis à jour, 
        # ne pas utiliser ces variables ensuite.
        old_update = e['update_function']
        
        # Affichage de la proselfprésence
        z,zmin,zmax = self.pred.rewind(self.sigpos).estimate(self.colname)
        f.add_trace(go.Scatter(x=self.pred.df.index, y=z, opacity=0.7,
                               name='pred',
                               line=dict(color='darkgreen', 
                                         dash='dot',
                                         width=1)), 
                    row=1, col=1)
        f.add_trace(go.Scatter(x=self.pred.df.index, y=zmin, opacity=0.7,
                               name='tubemin',
                               stackgroup='tube',
                               fill = 'none',
                               line=dict(color='green', 
                                         width=0)), 
                    row=1, col=1)
        f.add_trace(go.Scatter(x=self.pred.df.index, y=zmax-zmin, opacity=0.7,
                               name='tubemax',
                               stackgroup='tube',
                               fillcolor='rgba(0,180,0,0.5)',
                               line=dict(color='green', 
                                         width=0)), 
                    row=1, col=1)
            
        # =================================================================
        # ---- Begin: Callback Interactive  ----
        def update_plot(colname, sigpos):
            """ Mise à jour de l'affichage.
            """
            old_update(colname,sigpos) # met à jour les positions.
            
            pred = self.pred.rewind(self.sigpos)
            print('Ask for', self.colname)
            z,zmin,zmax = pred.estimate(self.colname)
            f.update_traces(selector=dict(name='pred'),
                            x = pred.df.index,
                            y = z)
            f.update_traces(selector=dict(name='tubemin'),
                            x = self.pred.df.index,
                            y = zmin)
            f.update_traces(selector=dict(name='tubemax'),
                            x = self.pred.df.index,
                            y = zmax-zmin)
        # ---- End: Callback Interactive ----

        # On remplace la fonction d'update (que l'on avait d'abord copiée).
        e['update_function'] = update_plot 
        return e

    
###########################################################################
#%% Un affichage interactif permettant de visualiser le contenu de la liste.
class Tube(Opset):
    """ L'Opset contenant les données à utiliser pour l'apprentissage.
    """
    
    def __init__(self, storename, phase=None, pos=0, name=""):
        """ Initialise les listes d'instants et d'opération."""
        super().__init__(storename, phase, pos, name)
        
        self.variables = set([self.df.columns[0]])
        self.factors = set(self.df.columns)
        self._reg = dict()
        self._sumlen = 0
        self.learn_params = dict(retry_number = 10,
                                 keep_best_number = 5,
                                 samples_percent = 0.01,
                                 max_features = 5)
        self.feature_params = dict(local_value='Absolute',
                                   use_time='No')
        self.tube_params = dict(tube_factor = 10.0,
                                filter_width = 20) 
            
            
    def __repr__(self):
        """ Affiche le nombre de sélections."""

        return "{}\n" \
               "TUBE : on {} variables.".format(Opset.__repr__(self),
                                                len(self.variables))
    
    
    def build_tube(self, colname, progress_bar=None):
        """ Création d'un tube pour une colonne."""
        
        retry_number = self.learn_params['retry_number']
        samples_percent = self.learn_params['samples_percent']
        max_features = self.learn_params['max_features']
        keep_best_number = self.learn_params['keep_best_number']
        
        reg_pop = dict()
        columns = self.factors
        cols = [c for c in columns if c != colname]
        if self.feature_params['use_time'] == 'Yes':
            cols += ['TIME']
        if self.feature_params['local_value'] == 'Median':
            cols += ['MEDIAN']
        if self.feature_params['local_value'] == 'Causal':
            cols += ['CAUSAL']
        nbcols = len(cols)
         
        miss = 0
        N = 0
        for i in range(retry_number):
            if progress_bar:
                progress_bar.value += 1
                
            # Choix des facteurs
            cc = np.random.permutation(cols)
            n = np.random.permutation(nbcols)+1
            n = np.min([n[0],max_features,nbcols])
            cc = cc[:n]
            X1 = []
            Y1 = []
            X2 = []
            Y2 = []
            i0 = 0
            for df in self:
                n = len(df)
                if 'TIME' in cc:
                    df['TIME'] = np.arange(n) # On rajoute le temps
                if 'MEDIAN' in cc:
                    df['MEDIAN'] = df[colname].median()
                if 'CAUSAL' in cc:
                    df['CAUSAL'] = df[colname][0]
                if i==0:
                    N += n # Comptage des longueur.
                    
                # Choix des observations
                pos1 = np.random.choice(np.arange(n),int(np.ceil(n*samples_percent)))
                #pos2 = np.random.choice(np.arange(n),int(np.ceil(n*samples_percent)))
                pos2 = np.delete(np.arange(n),pos1)
                pos2 = np.random.choice(pos2,int(np.ceil(n*samples_percent)))
                
                df1 = df.iloc[pos1]
                df2 = df.iloc[pos2]
                x1 = df1[cc]
                y1 = df1[colname]
                x2 = df2[cc]
                y2 = df2[colname]
                x1.index = range(i0,i0+len(x1))
                y1.index = range(i0,i0+len(x1))
                x2.index = range(i0,i0+len(x2))
                y2.index = range(i0,i0+len(x2))
                i0 = i0+len(x1)
                Y1.append(y1)
                X1.append(x1)
                Y2.append(y2)
                X2.append(x2)
                
            dfx1 = pd.concat(X1)
            dfy1 = pd.concat(Y1)
            dfx2 = pd.concat(X2)
            dfy2 = pd.concat(Y2)
            if i==0:
                self._sumlen = N
                
            # Création du modèle.
            reg = linear_model.LinearRegression()
            reg = reg.fit(dfx1,dfy1)
            r2 = reg.score(dfx2,dfy2)
            if i<keep_best_number:
                reg_pop[i] = (reg,cc,r2)
            else:
                R2 = np.array([reg_pop[i][2] for i in reg_pop])
                ind = R2.argsort()
                if r2>R2[ind[0]]:
                    reg_pop[ind[0]] = (reg,cc,r2)
                    miss = 0
                else:
                    miss += 1
                    if miss==keep_best_number:
                        if progress_bar:
                            progress_bar.value += retry_number-i
                        break
        
        return reg_pop.values()
        
        
    def fit(self, progress_bar=None, message_label=None):
        """ Apprentissage d'un modèle.
        
            Le modèle est appris uniquement sur les variables et facteurs
            sélectionnés.
            
            :return: le Tube appris.
        """
        
        if len(self) == 0:
            raise OpsetError(self.storename,"No data")
        
        sigpos = self.sigpos 

        if progress_bar:
            retry_number = self.learn_params['retry_number']
            progress_bar.max = len(self.variables)*retry_number+1
            progress_bar.value = 0
            
        # Création des régressions.
        for colname in self.variables:
            if message_label:
                message_label.value = "Working on target " + colname + " ..."
            self._reg[colname] = self.build_tube(colname, progress_bar=progress_bar)
            
        if progress_bar:
            progress_bar.value = 0
        if message_label:
            message_label.value = ""
        return self.rewind(sigpos)
    
    
    def estimate(self, name=None):
        """ Renvoie la prédiction et le tube de confiance.
        
            :return: un tableau numérique contenant les valeurs prédites.
        """

        if name is None:
            colname = self.colname
        else:
            colname = get_colname(self.df.columns,name)
            
        df = self.df
        if colname not in self._reg or len(df)==0:
            y = df[colname].values
            z = np.zeros(y.shape)
            z.fill(np.nan)
            zmin = zmax = z
        
        else:
            Z = np.array([])
            reg_list = self._reg[colname]
            Z = np.ndarray((0,len(df)))
            df['TIME'] = np.arange(len(df))
            df['MEDIAN'] = df[colname].median()
            df['CAUSAL'] = df[colname][0]
            for reg, cols, r2 in reg_list:
                x = df[cols]
                z = reg.predict(x)
                Z = np.vstack((Z,z))
            del df['CAUSAL']
            del df['MEDIAN']
            del df['TIME']
            z = Z.mean(axis=0)
            zmax = Z.max(axis=0)
            zmin = Z.min(axis=0)
            q = self.tube_params['tube_factor']
            w = self.tube_params['filter_width']
            if w>0:
                zmin = signal.savgol_filter(z-q*(z-zmin),
                                        window_length=2*w+1,
                                        polyorder=2,
                                        deriv=0)
                zmax = signal.savgol_filter(z+q*(zmax-z),
                                        window_length=2*w+1,
                                        polyorder=2,
                                        deriv=0)
            else:
                zmin = z-q*(z-zmin)
                zmax = z+q*(zmax-z)
                
        return z,zmin,zmax
    
    
    def describe(self):
        """ Description des estimations.
        
            :return: un DataFrame avec les cibles sur chaque ligne et les
                     facteurs en colonnes.
        """
        desc = pd.DataFrame(0, columns=list(self.df.columns)+['TIME','MEDIAN','CAUSAL'],
                            index=list(self._reg.keys()))
        for colname in self._reg:
            reg_list = self._reg[colname]
            for reg,cc,r2 in reg_list:
                for c in cc:
                    desc[c].loc[colname] += 1
        desc.index.name = "Regresions"
        return desc
        
     # ============================= Scores ===============================
    def local_scores(self):
        """ Renvoie les scores de toutes les variables du signal courant."""
        
        scr = pd.DataFrame(0, index=[self.current_record()],
                           columns=["N"]+list(self._reg.keys()))
        
        scr["N"][self.current_record()] = len(self.df)
        for colname in self._reg:
            z, zmin, zmax = self.estimate(colname)
            y = self.df[colname].values
            s = sum(y>zmax)+sum(y<zmin)
            scr[colname][self.current_record()] = s
            
        return scr
    
    
    def scores(self):
        """ Renvoie le score global du signal courant."""
        
        scr = pd.DataFrame(0, index=self.records,
                           columns=["N"]+list(self._reg.keys()))
        
        for df in self:
            scr["N"][self.current_record()] = len(df)
            for colname in self._reg:
                z, zmin, zmax = self.estimate(colname)
                y = self.df[colname].values
                s = sum(y>zmax)+sum(y<zmin)
                scr[colname][self.current_record()] = s

        return scr
    
    
    def plot_scores(self):
        """ Affichage des scores."""

        scr = self.scores()
        f = go.Figure()
        for colname in scr.columns[1:]:
            f.add_trace(go.Bar(name = colname, 
                               x = scr.index, 
                               y = scr[colname]/scr["N"]))
        f.update_layout(barmode='stack',
                        title="Out of tube proportions")
 
        return f

    
    # ========================== Affichages ===============================
    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Création de l'interface graphique.
        
            Un tube est ajouté autour des données. Une ligne pointillée 
            verte dans le tube correspond à l'estimation.
            
            todo::  améliorer la gestion du 'range' de la figure qui est 
                    altérée car il est difficile de créer le tube sans 
                    utiliser un stackgroup. Une solution serait de
                    transformer l'index et de faire une boucle 'toself'.
        """

        # Récupération de l'interface de l'Opset.
        e = Opset.make_figure(self,f,phase,pos,name)
        
        
        # =================================================================
        old_update = e['update_function']
        
        # Affichage de la proba de présence
        z,zmin,zmax = self.estimate()
        f.add_trace(go.Scatter(x=self.df.index, y=z, opacity=0.7,
                               name='pred',
                               line=dict(color='darkgreen',
                                         dash='dot',
                                         width=1)), 
                    row=1, col=1)
        f.add_trace(go.Scatter(x=self.df.index, y=zmin, opacity=0.7,
                               name='tubemin',
                               stackgroup='tube',
                               fill = 'none',
                               #fill='toself',
                               #fillcolor='rgba(0,180,0,0.5)',
                               line=dict(color='green', 
                                         width=0)), 
                    row=1, col=1)
        f.add_trace(go.Scatter(x=self.df.index, y=zmax-zmin, opacity=0.7,
                               name='tubemax',
                               stackgroup='tube',
                               #fill='toself',
                               fillcolor='rgba(0,180,0,0.5)',
                               line=dict(color='green', 
                                         width=0)), 
                    row=1, col=1)
        if len(z)>0:
            z0 = zmin.min()
            z1 = zmax.max()
            f.update_yaxes(range=(z0-0.1*(z1-z0),z1+0.1*(z1-z0)))
            
        # ---- Begin: Callback Interactive  ----
        def update_plot(colname, sigpos):
            """ Mise à jour de l'affichage.
            """
            old_update(colname,sigpos) # met à jour les positions.
            
            z,zmin,zmax = self.estimate()
            f.update_traces(selector=dict(name='pred'),
                            x = self.df.index,
                            y = z)
            f.update_traces(selector=dict(name='tubemin'),
                            x = self.df.index,
                            y = zmin)
            f.update_traces(selector=dict(name='tubemax'),
                            x = self.df.index,
                            y = zmax-zmin)
            if len(z)>0:
                z0 = zmin.min()
                z1 = zmax.max()
                f.update_yaxes(range=(z0-0.1*(z1-z0),z1+0.1*(z1-z0)))
        
            
        # On remplace la fonction d'update (que l'on avait d'abord copiée).
        e['update_function'] = update_plot 
        # ---- End: Callback Interactive ----
        
        # =================================================================
        # --------- Liste de sélection des variables à prédire ------------
        wlmv = widgets.SelectMultiple(options = self.df.columns,
                                      value = tuple(self.variables),
                                      description = 'Targets',
                                      rows=8,
                                      disabled = False,
                                      layout=widgets.Layout(width='250px'))

        def auto_update_variable(*args):
            self.variables.add(e['variable_dropdown'].value)
            wlmv.value = tuple(self.variables)
        
        def update_variables(*args):
            self.variables = set(wlmv.value)
            
        e['variable_selection'] = wlmv
        e['variable_dropdown'].observe(auto_update_variable,'value')
        wlmv.observe(update_variables,'value')
        # -------- Fin de liste ---------
        
         # -------- Liste de sélection des facteurs de prdiction ----------
        wlmf = widgets.SelectMultiple(options = self.df.columns,
                                      value = tuple(self.factors),
                                      description = 'Factors ',
                                      rows=8,
                                      disabled = False,
                                      layout=widgets.Layout(width='250px'))

        def update_factors(*args):
            self.factors = set(wlmf.value)
        
        e['factor_selection'] = wlmf
        wlmf.observe(update_factors,'value')
        # -------- Fin de liste ---------
        
        # =================================================================
        # --------- Barre de progression  ------------
        wp = widgets.IntProgress(value=0, min=0, max=10, step=1,
                                 description='Progress:',
                                 bar_style='', # 'success','info','warning','danger',''
                                 orientation='horizontal',
                                 layout=widgets.Layout(width='500px'))      
        e['progress_bar'] = wp
        
        wml = widgets.Label(value="")
        e['message_label'] = wml
        # --------- Fin de progression --------

        # =================================================================
        # Boutton pour l'apprentissage.
        wbl = widgets.Button(description='Learn')
        if self._reg:
            wbl.description = 'Relearn'
            wbl.button_style = 'success'    
        else:
            wbl.description = 'Learn'
            wbl.button_style = 'info'
        
        # ---- Callback ----
        def wbl_on_click(b):
            """ Callbacks du boutton d'apprentissage."""
            self.variables = set(wlmv.value)
            
            b.button_style = 'warning'
            b.description = 'Learning...'
            self.fit(progress_bar=wp, message_label=wml)
            update_plot(self.colname,self.sigpos)
            b.description = 'Relearn'
            b.button_style = 'success'
        # ---- Callback ----

        wbl.on_click(wbl_on_click)
        
        e['learn_button'] = wbl
        # -------- Fin de l'apperntissgae --------
        
        return e

    
    def param(self):
        """ Interface de paramétrage.
        
            Cette fonction affiche dans une cellule de dialogue les paramètres
            courants et offre la possibilité de les modifier graphiquement.
        """
        
        def update_parameters(retry, keep, sample, features, factor, width, localv, utime):
            
            self.learn_params['retry_number'] = retry 
            self.learn_params['keep_best_number'] = keep
            self.learn_params['samples_percent'] = sample 
            self.learn_params['max_features'] = features
            self.tube_params['tube_factor'] = factor
            self.tube_params['filter_width'] = width
            self.feature_params['local_value'] = localv
            self.feature_params['use_time'] = utime
          
        # =================================================================
        # Widgets pour les indicateurs.
        wtrn = widgets.IntText(description = "Retry number", 
                               value = self.learn_params['retry_number'],
                               layout=widgets.Layout(width='250px'))
        wtkb = widgets.IntText(description = "Final population", 
                               value = self.learn_params['keep_best_number'],
                               layout=widgets.Layout(width='250px'))
        wtsp = widgets.FloatText(description = "Sample percentage", 
                                 value = self.learn_params['samples_percent'],
                                 layout=widgets.Layout(width='250px'))
        wtmf = widgets.IntText(description = "Maximum features", 
                               value = self.learn_params['max_features'],
                               layout=widgets.Layout(width='250px'))
        
        # Widgets pour les caractéristiques spéciales.
        wsmc = widgets.Select(options=['Absolute', 'Median', 'Causal'],
                              value=self.feature_params['local_value'],
                              description='Local value',
                              rows=3,
                              layout=widgets.Layout(width='250px'))
        wsmt = widgets.Select(options=['Yes', 'No'],
                              value=self.feature_params['use_time'],
                              description='Use time',
                              rows=2,
                              layout=widgets.Layout(width='250px'))

        # Widgets pour les tubes.
        wttt = widgets.FloatText(description = "Tube size", 
                                 value = self.tube_params['tube_factor'],
                                 layout=widgets.Layout(width='250px'))
        wttw = widgets.IntText(description = "Filter width", 
                                 value = self.tube_params['filter_width'],
                                 layout=widgets.Layout(width='250px'))

        fbox = widgets.VBox([widgets.Label("Learning parameters (.learn_params):"),
                             widgets.HBox([wtrn, wtkb]),
                             widgets.HBox([wtsp, wtmf]),
                             widgets.Label("Feature parameters (.feature_params):"),
                             widgets.HBox([wsmc, wsmt]),
                             widgets.Label("Tube parameters (.tube_params:)"),
                             widgets.HBox([wttt, wttw])
                            ])

        out = widgets.interactive(update_parameters,
                                  retry=wtrn, keep=wtkb,
                                  sample=wtsp, features=wtmf,
                                  factor=wttt, width=wttw,
                                  localv=wsmc, utime=wsmt)
        
        return fbox
        
    
    def plot(self,phase=None,pos=None,name=None):
        """ On ajoute à l'affichage de l'Opset une sélection d'instants."""
        f = make_subplots(rows=1, cols=1,
                          specs=[[{"type": "scatter"}]])
        f = go.FigureWidget(f)
        e = self.make_figure(f,phase,pos,name)
        out = widgets.interactive(e['update_function'], 
                                  colname=e['variable_dropdown'], 
                                  sigpos=e['signal_slider'])
        
        bxplot = widgets.VBox(
            [widgets.HBox([e['variable_dropdown'], 
                           e['previous_button'], e['next_button']]),
             widgets.HBox([f, e['signal_slider']])
            ])
        # Pour info :
        # f = tabs.children[0].children[1].children[0]
        
        bxlearn = widgets.VBox([e['progress_bar'], 
                                widgets.HBox([widgets.VBox([e['variable_selection'],
                                                            e['learn_button']]),
                                              widgets.VBox([e['factor_selection'],
                                                            e['message_label']])
                                             ])
                               ])
        
        bxparam = self.param()
        
        tabs = widgets.Tab(children = [bxplot, bxparam, bxlearn])
        tabs.set_title(0, "Plot")
        tabs.set_title(1, "Param")
        tabs.set_title(2, "Learn")
        return tabs

    
