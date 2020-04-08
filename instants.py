# -*- coding: utf-8 -*-
"""
INSTANTS - Extracteur d'instants.

Des fonctions de gestion de l'extracteur d'instants.

todo:: 

- [ ] Modifier `make_indicators` pour ne pas refaire les calculs après ajout
        de nouveaux éléments.
- [ ] Vérifier que le `load` dispose des variables utiles dans le `_clf`.
- [ ] Faire un calcul de synthèse avec régression logistique finale.

**Versions**

1.0.3 - Algorithmes.


Created on Fri March 27 19:27:00 2020

@author: Jérôme Lacaille
"""

__date__ = "2020-03-29"
__version__ = '1.0.2'

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal
from sklearn import tree
from tabata.opset import Opset, nameunit, OpsetError


###########################################################################
#%% Fonctions auxiliaires.
def indicator(y,width,order,sigma,deg=2):
    """ Création d'un indicateur de comptage de bosses.
    
        Le but est de remplacer un signal par un indicateur qui donne pour 
        chaque instant la position entre deux bosses ou creux successifs.

        Elle commence par lisser le signal avec le filtre polynomial de 
        Savitsky-Golay pour un polynome d'ordre 2 (ou deg).
        Le principe est de pouvoir extraire les pentes et courbures du 
        signal.
        On regarde alors les passages par zéro.
        Cependant, un passage par zéro n'est pas forcément précis, 
        on va donc se donner une marge au dessus et une autre marge au 
        dessous.
        La marge est un multiple de l'écart-type du signal lissé. Cemme 
        cette marge doit être commune à tous les signaux on la calcule à
        l'extérieur.

          z,k = indicator(y,width,order,sigma,deg)

        :param y: le signal d'entrée.
        :param width: la largeur de bande du filtre.
        :param order: l'ordre de dérivation (1 ou 2).
        :param sigma: un seuil de détection de passage par zéro.

        Si le seuil de détection est négatif on étudie les sorties de zéros
        en dessous du seuil.

        :return z: l'indicateur calculé.
        
        todo:: faire un calcul global de epsilon par variable et niveau de
        lissage.
    """
    x = signal.savgol_filter(y,window_length=width,polyorder=2,deriv=order)
    
    if sigma>0:
        b = x>sigma
    else:
        b = x<sigma
        
    dp = np.diff(b.astype(int))
    k = list(np.argwhere(dp).ravel())
    z = np.zeros(y.shape)
    i0 = 0
    if len(k) >= 1:
        z0 = 1.0 - float(dp[k[0]]==1)
        for i in k+[len(y)]:
            z[i0:i] = np.linspace(z0,z0+1.0,i-i0)
            z0 = z0+1.0
            i0 = i
    return z


###########################################################################
#%% Un affichage interactif permettant de visualiser le contenu de la liste.
class Selector(Opset):
    """ Interface de sélection et d'apprentissage du détecteur d'instant.
    
        Un affichage interactif utilisant plotly qui permet de
        sélectionner des points dans une liste d'observations.
        
        Le selecteur va créer deux objets :
        
        * `sel_instants`: un dictionnaire indéxé par un numéro d'opération
                          qui contient:
                          - le nom de la variable vue, 
                          - la position réelle sélectionnée, 
                          - la position dans le signal.
        * `op_viewed`:    la liste des signaux vraiment observés depuis la 
                          création du Selector.
                          
        Maintenir la liste des signaux vraiment observés est une information
        qui permet d'utiliser des données vues par l'utilisateur, mais qu'il
        n'a pas jugé utile de sélectionner. Cela donnera une pondération à
        la vaildation de l'algorithme.
        
        La position réelle sélectionnée est la position sur la courbe
        sélectionnée. Elle peut être différente de la position dans le
        signal si par exemple une pahse est mise en évidence. Dans ce cas,
        en cliquant sur la phase, on obtiendra la position au sein de cette
        phase qui est nécessairement plus petite.
        
        todo:: Recréer un classifieur avec juste les indicateurs nécessaires.
    """

    def __init__(self, storename, phase=None, pos=0, name=""):
        """ Initialise les listes d'instants et d'opération."""
        Opset.__init__(self, storename, phase, pos, name)
        self.selected = dict()
        self.viewed = set()
        self.computed = dict()
        self.idcodes = [] # Liste des indicateurs conservés.
        
        self._dsi = None # Un pointeur vers un sélecteur contenant les indicateurs.
        self._clf = None # Le classifieur une fois créé par `fit()`
        
        
    def __repr__(self):
        """ Affiche le nombre de sélections."""
        ops = len(self.viewed)
        pts = len(self.selected)
        return "{}\n" \
               "{} instant(s) sélectionnés parmis {} courbes " \
               "observées".format(Opset.__repr__(self), pts, ops)

    
    def clear_selection(self):
        """ Réinitialise la liste des sélections et observations."""
        self.viewed.clear()
        self.selected.clear()
        self.computed.clear()
        
    
    # ================== Apprentissage ====================================
    def make_indicators(self, 
                        idfilename=None,
                        range_width = None,
                        range_sigma = None,
                        max_order=2):
        """ Construction de l'Opset des indicateurs."""

        if not self.selected:
            raise OpsetError(self.storename,"Nothing to learn !")
            
        # Nom du fichier de sauvegarde
        if idfilename is None:
            i = self.storename.find('.')
            idfilename = self.storename[:i] + '_I' + self.storename[i:]
        
        # Liste des variables
        colnames = {self.selected[s][0] for s in self.selected}
        # Liste des observations et instants selectionnés
        obs = [s for s in self.selected]
        ind = [self.selected[s][1] for s in self.selected]
        
        # Mémorisation des données utiles dans une liste data.
        data = [] # Une liste de matrices de signaux.
        L = []    # Les longueurs de chaque signal.
        for df in self.iterator(obs):
            L = L +[len(df)]
            data.append(df[colnames])
            
        # Demi largeur minimale du filtre.
        if range_width is None:
            L0 = max(10,int(np.floor(np.min(L)/100)))
            range_width = range(L0,L0*10+1,L0)
        
        # Ordre de dérivation.
        deg_poly = max(2,max_order)
        
        # Calcul des epsilons.
        epsilon = dict()
        for l in range_width:
            w = 2*l+1 # La largeur doit être impaire pour SG.
            R = np.ndarray((max_order,len(colnames)))
            for d in range(max_order):
                for i,colname in enumerate(colnames):
                    r = []
                    k = 0
                    for df in data:
                        a = df[colname]
                        b = signal.savgol_filter(a,
                                                 window_length=w,
                                                 polyorder=deg_poly,
                                                 deriv=d+1)
                        c = signal.savgol_filter(b,
                                                 window_length=w,
                                                 polyorder=deg_poly,
                                                 deriv=0)
                        r = r + [np.std(b-c)]
                    R[d,i] = max(r)
            epsilon[w] = R
        
        # Nombre de sigma
        if range_sigma is None:
            range_sigma = range(5,26,10)
            
        # Calcul des indicateurs.
        idcodes = []
        first = True
        dsi = Selector(idfilename)
        dsi.clean()
        for j,df in enumerate(data):
            #C = np.ndarray((0,len(df)))
            a = np.arange(0,len(df))
            C = np.vstack((a,np.flip(a), a/(len(df)-1)))
            if first:
                idnames = ['LEN[pts]','REV[pts]','PERCENT[%]']
                idcodes = [('LEN',0,0,0,0.0), ('REV',0,0,0,0.0), ('PERCENT',0,0,0,0.0)]
            for i,colname in enumerate(colnames):
                a = df[colname].values
                name,unit = nameunit(colname)
                for l in range_width:
                    w = 2*l + 1
                    for d in range(max_order):
                        eps = epsilon[w][d,i]
                        for s in range_sigma:
                            for e in [1, -1]:
                                c = indicator(a,w,d+1,e*s*eps,deg_poly)
                                C = np.vstack((C,c))
                                if first: # len(k)>0 and k not in K:
                                    code = 'w{}o{}'.format(l, d+1)
                                    if e>0:
                                        code = code + 'u{}'.format(s)
                                    else:
                                        code = code + 'd{}'.format(s)
                                    idnames = idnames +\
                                         ["{}[{}]".format(name,code)]
                                    idcodes = idcodes +\
                                        [(colname, l, d, e*s, eps)]
            first = False
            dfi = pd.DataFrame(C.T, index=df.index, columns=idnames)
            self.idcodes = idcodes # On sauvegarde les indicateurs.
            dsi.put(dfi) # Le nm de l'opération est dans l'index.
        
        # Pour visualiser les données sur les signaux 
        # on conserve les instants sélectionnés.
        dsi.computed = dict(enumerate(zip(['__all__']*len(ind), ind)))
        dsi.rewind()
        dsi.idcodes = self.idcodes
        self._dsi = dsi
        return dsi
    
    
    def fit(self,
            retry_number=1, 
            retry_percentile=80,
            samples_percent=0.01, 
            min_samples_split=0.05):
        """ Apprend un arbre de décision basé sur les indicateurs.
            
            :param percent: le pourcentage de points tirés par signal pour
                            l'apprentissage.
            :param min_sample_split: paramètre extrait de DecisionTreeClassifier
                            qui spécifie le nombre minimal d'observation par feuille.
        """
        
        if self._dsi is None:
            print("Building indicators ...")
            dsi = self.make_indicators()
        else:
            dsi = self._dsi
        
        # Fabrication des données d'apprentissage.
        def find_best_parameters(p,split,cols=None):
            X = []
            Y = []
            i0 = 0
            for df in dsi.iterator():
                n = len(df)
                var,ind = dsi.computed[dsi.sigpos]
                pos = np.random.choice(np.arange(n),int(np.ceil(n*p)))
                if cols:
                    x = df.iloc[pos,cols]
                else:
                    x = df.iloc[pos]
                x.index = range(i0,i0+len(x))
                y = pd.DataFrame(1 - 2*(pos<=ind), index=x.index, columns=['TARGET'])
                i0 = i0+len(x)
                Y.append(y)
                X.append(x)
            dfx = pd.concat(X)
            dfy = pd.concat(Y)

            # Création du modèle.
            clf = tree.DecisionTreeClassifier(min_samples_split=split)
            clf = clf.fit(dfx,dfy)

            return clf
    
    
        fi = np.zeros(len(dsi.df.columns))
        for k in range(retry_number):
            clf =  find_best_parameters(samples_percent,min_samples_split)
            fi += clf.feature_importances_
        seuil = np.percentile(fi,retry_percentile)
        newcodes = {i:c for i,c in enumerate(dsi.idcodes) if fi[i]>seuil}
        self.idcodes = list(newcodes.values())
        
        # Apprentissage final.
        print("Keeping {} indicators over {}".format(len(self.idcodes),len(dsi.idcodes)))
        keepcols = list(newcodes.keys())
        clf = find_best_parameters(min(0.5,samples_percent*retry_number),
                                   min_samples_split,
                                   keepcols)
        self._clf = clf

        return clf
    
    
    def belief(self,filter_width=100):
        """ Calcul d'un indicateur de présomption de détection."""
        
        df = self.df
        if self._clf is None:
            return np.zeros(df.index.shape)
        clf = self._clf
        
        # On crée les indicateurs.
        C = np.ndarray((0,len(df)))
        a = np.arange(0,len(df))
        for colname, l, d, es, eps in self.idcodes:
            if l==0:
                if colname=="LEN":
                    c = a
                elif colname == "REV":
                    c = np.flip(a)
                elif colname == "PERCENT":
                    c = a/(len(df)-1)
            else:
                a = df[colname].values
                w = 2*l + 1
                deg_poly = max(2,d)
                c = indicator(a,w,d+1,es*eps,deg_poly)
            C = np.vstack((C,c))
            
        ip = clf.predict(C.T) # Eventuellement pondérer.
        p = signal.savgol_filter(ip,
                                 window_length=2*filter_width+1,
                                 polyorder=2,
                                 deriv=1)
        
        p = np.maximum(p,0)
        Z = p.sum()
        if Z == 0.0:
            Z=1.0
        p /= Z
        
        mx = np.argmax(p)
        self.computed[self.sigpos] = (self.colname,mx)
        
        return p
                     
    
    def load(self,filename):
        """ Recharge un nouveau fichier à analyser."""
        ds = Selector(filename,self.phase)
        ds.idcodes = self.idcodes
        ds._clf = self._clf
        
        return ds
        
    # ====================== Affichages ===========================                              
    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Création de l'interface graphique."""

        # Récupération de l'interface de l'Opset.
        e = Opset.make_figure(self,f,phase,pos,name)
        
        # self.sigpos et self.colname sont mis à jour, 
        # ne pas utiliser ces variables ensuite.
        old_update = e['update_function']
        
        # Affichage de la proba de présence
        p = self.belief()
        f.add_trace(go.Scatter(x=self.df.index, y=p), row=2, col=1)

        f.update_yaxes(domain=(0.315, 1.0), row=1, col = 1)
        f.update_yaxes(domain=(0.0, 0.285), row=2, col = 1)
        f.layout.xaxis2.titlefont.color = "blue"
            
            
        # =================================================================
        # ---- Begin: Callback Interactive  ----
        def update_plot(colname, sigpos, width):
            """ Mise à jour de l'affichage.
                
                On rajoute des barres verticales pour identifier les 
                instants sélectionnés.
            """
            old_update(colname,sigpos)
        
            # Mise à jour des probas.
            f.update_traces(x=self.df.index, y=self.belief(width), row=2)
            
            self.viewed.add(sigpos)
            f.layout.shapes = []
            if self.sigpos in self.selected:
                i = self.selected[self.sigpos][1]
                x0 = self.df.index[i]
                y0 = min(self.df[colname])
                y1 = max(self.df[colname])
                f.add_shape(type='line',
                                  x0=x0, y0=y0, x1=x0, y1=y1,
                                  line= {'color': 'rgb(171, 50, 96)',
                                        'width': 2,
                                        'dash': 'dashdot'},
                                  row=1, col=1)
            if self.sigpos in self.computed:
                i = self.computed[self.sigpos][1]
                x0 = self.df.index[i]
                y0 = min(self.df[colname])
                y1 = max(self.df[colname])
                f.add_shape(type='line',
                                  x0=x0, y0=y0, x1=x0, y1=y1,
                                  line= {'color': 'rgb(96, 50, 171)',
                                        'width': 1,
                                        'dash': 'dot'},
                                  row=1, col=1)
        # ---- End: Callback Interactive ----
        
        # =================================================================
        # Widgets pour l'apprentissage.
        wtrn = widgets.IntText(description = "Retry", value = 10,
                               layout=widgets.Layout(width='140px'))
        wtrp = widgets.IntText(description = "Percentile", value = 80,
                              layout=widgets.Layout(width='140px'))
        wtsp = widgets.FloatText(description = "Sample", value = 0.01,
                                layout=widgets.Layout(width='140px'))
        wtss = widgets.FloatText(description = "Split", value = 0.05,
                                layout=widgets.Layout(width='140px'))
                
        # =================================================================
        # Boutton pour l'apprentissage.
        wbl = widgets.Button(description='Learn')
        if self.selected:
            if self._clf:
                wbl.description = 'Relearn'
                wbl.button_style = 'success'    
            else:
                wbl.description = 'Learn'
                wbl.button_style = 'info'
        else:
            wbl.description = 'No Target'
            wbl.button_style = 'danger'        
        
        # ---- Callback ----
        def wbl_on_click(b):
            """ Callbacks du boutton d'apprentissage."""
            b.button_style = 'warning'
            b.description = 'Learning...'
            if not self.selected:
                b.description = 'No Target'
                b.button_style = 'danger'
            else:
                self.fit(wtrn.value, wtrp.value, wtsp.value, wtss.value)
                update_plot(self.colname,self.sigpos,wf.value)
                b.description = 'Relearn'
                b.button_style = 'success'
        # ---- Callback ----

        wbl.on_click(wbl_on_click)

        # =================================================================
        # Sélection d'un point sur la courbe.
        # ---- Calback ----
        def selection_fn(trace, points, selector):
            """ Le callback qui permet de sélectionner un instant.
            """
            if len(points.point_inds)>0:
                i0 = points.point_inds[0]
                x0 = trace.x[i0]
                scatter1 = trace.parent.data[0]
                x1 = scatter1.x
                i1 = int(min(np.argwhere(x1>=x0)))
                y0 = min(scatter1.y)
                y1 = max(scatter1.y)
                trace.parent.layout.shapes = [{
                    'type': 'line',
                    'x0': x0,
                    'y0': y0,
                    'x1': x0,
                    'y1': y1,
                    'line': {'color': 'rgb(171, 50, 96)',
                    'width': 2,
                    'dash': 'dashdot'}}]
                self.selected[self.sigpos] = (self.colname, i1)
            if self.selected:
                if self._clf:
                    wbl.description = 'Relearn'
                    wbl.button_style = 'success'    
                else:
                    wbl.description = 'Learn'
                    wbl.button_style = 'info'
            else:
                wbl.description = 'No Target'
                wbl.button_style = 'danger'        
        # ---- Callback ----
        
        # Début de l'affichage.
        scatter = f.data[0]
        scatter.on_click(selection_fn)
        if phase is not None:
            scatter2 = f.data[1]
            scatter2.on_click(selection_fn)
        
        
        # =================================================================
        # Slider pour la largeur du filtre de croyance.
        wf = widgets.IntSlider(value=10, min=10, max=500, step=10,
                               orientation='horizontal',
                               description='Filter',
                               continuous_update=False,
                               layout=widgets.Layout(width='400px'))
           
            
        # Afficher la barre si un signal est présent.
        update_plot(self.colname,self.sigpos,wf.value)

        # On remplace la fonction d'update (que l'on avait d'abord copiée).
        e['update_function'] = update_plot 
        e['filter_slider'] = wf
        e['learn_button'] = wbl
        e['learn_param'] = [wtrn, wtrp, wtsp, wtss]
        return e
    
    
    def plot(self,phase=None,sigpos=None,colname=None):
        """ On ajoute à l'affichage de l'Opset une sélection d'instants."""
        f = make_subplots(rows=2, cols=1,
                          shared_xaxes=True,
                          vertical_spacing=0.03,
                          specs=[[{"type": "scatter"}],
                                [{"type": "scatter"}]])
        f = go.FigureWidget(f)
        e = self.make_figure(f,phase,sigpos,colname)
        out = widgets.interactive(e['update_function'], 
                                  colname=e['variable_dropdown'], 
                                  sigpos=e['signal_slider'],
                                  width=e['filter_slider'])
        
        boxes = widgets.VBox(
            [widgets.HBox([e['variable_dropdown'], 
                          e['previous_button'], e['next_button']]),
             widgets.HBox([f, e['signal_slider']]),
             widgets.HBox([e['filter_slider'], e['learn_button']]),
             widgets.HBox(e['learn_param'])
            ])
        
        return boxes