# -*- coding: utf-8 -*-
"""
INSTANTS - Extracteur d'instants.

Des fonctions de gestion de l'extracteur d'instants.

**Versions**

1.0.3 - Algorithmes.
1.0.4 - Mise à jour de la doc.
1.0.5 - Indicateurs rétrogrades et découpage L/R.
1.0.6 - L'apprentissage retourne self.
        Réinstallation du pointeur après extraction.
1.0.7 - Tabs pour l'apprentissage
1.0.8 - Limitation des indicateurs si l'isntant est proche d'un bord.
1.0.9 - Appel de super()


todo::
    - Choix des variables dans une liste à cocher.
    - Progress bar pendant l'apprentissage.
    
Created on Fri March 27 19:27:00 2020

@author: Jérôme Lacaille
"""

__date__ = "2020-05-10"
__version__ = '1.0.9'

import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal
from sklearn import tree
from .opset import Opset, OpsetError
from .plots import nameunit


###########################################################################
#%% Fonctions auxiliaires.
def indicator(y,width,order,sigma,deg=2):
    """ Création d'un indicateur de comptage de bosses et de creux.
    
        Le but est de remplacer un signal par un indicateur qui donne pour 
        chaque instant la position entre bosses ou creux successifs.

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

        :param y:     le signal d'entrée.
        :param width: la largeur de bande du filtre.
        :param order: l'ordre de dérivation (1 ou 2).
        :param sigma: un seuil de détection de passage par zéro.
        :param deg:   le dergé maximal du polynôme de lissage.

        Si le seuil de détection est négatif on étudie les sorties de zéros
        en dessous du seuil.

        :return z: l'indicateur calculé.
    """
    x = signal.savgol_filter(y,window_length=width,polyorder=deg,deriv=order)
    
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
        
        * `selected`:   un dictionnaire indéxé par un numéro d'opération
                        qui contient:
                          - le nom de la variable vue, 
                          - la position réelle sélectionnée, 
                          - la position dans le signal.
        * `viewed`:     la liste des signaux vraiment observés depuis la 
                        création du Selector.
        * `variables`:  la liste des variables observées par l'expert.
        * `computed`:   un dictionnaire contenant les instants calculés 
                        une fois le détecteur d'instants calibré.
        * `idcodes`:    la description des indicateurs utilisés.
        
        Deux conteneurs "cachés" :
        
        * `_dsi`:       un Selector contenant les indicateurs pour 
                        l'apprentissage.
        * `_clf`:       le classifieur finalement appris (un arbre de décision.)
                          
        Maintenir la liste des signaux vraiment observés est une information
        qui permet d'utiliser des données vues par l'utilisateur, mais qu'il
        n'a pas jugé utile de sélectionner. Cela donnera une pondération à
        la vaildation de l'algorithme.
        
        La position réelle sélectionnée est la position sur la courbe
        sélectionnée. Elle peut être différente de la position dans le
        signal si par exemple une phase est mise en évidence. Dans ce cas,
        en cliquant sur la phase, on obtiendra la position au sein de cette
        phase qui est nécessairement plus petite
        
        Le sélecteur dispose de trois jeux de paramètres qu'il est possible
        de modifier directement ou par l'appel des méthodes `param()` 
        et `plot()`.

        learn_params:
            retry_number     (10)   le nombre de modèles testés.
            retry_percentile (80)   un quantile minimum sur le taux d'apparition
                                    d'une variable dans les modèles testés
                                    pour la conserver.
            sample_percent   (0.01) le taux de sélection de points dans les 
                                    signaux pour l'apprentissage.
            min_sample_split (0.05) le taux minimal dans une règle pour autoriser
                                    un découpage.
                                    
        feature_params:
            range_width             les tailles de demi-fenêtres de lissage.
            range_sigma             les seuils pour la détecton du passage
                                    par zéro.
            max_order        (2)    le degré du polynome pour le lissage
                                    se Savitsky-Golay.
                                    
        predict_params:
            filter_width     (100)  la demi-taille du filtre de lissage des 
                                    prédictions.
    """

    def __init__(self, storename, phase=None, pos=0, name=""):
        """ Initialise les listes d'instants et d'opération."""
        super().__init__(storename, phase, pos, name)
        self.selected = dict()
        self.viewed = set()
        self.variables = set()
        self.computed = dict()
        self.idcodes = [] # Liste des indicateurs conservés.
        
        self._dsi = None # Un pointeur vers un sélecteur contenant les indicateurs.
        self._clf = None # Le classifieur une fois créé par `fit()`
        
        self.learn_params = dict(retry_number=10, 
                                 retry_percentile=80,
                                 samples_percent=0.01, 
                                 min_samples_split=0.05)
        
        self.feature_params = dict(range_width = None,
                                   range_sigma = range(5,26,10),
                                   max_order=2)
        
        self.predict_params = dict(filter_width=100)
        
        
    def __repr__(self):
        """ Affiche le nombre de sélections."""
        ops = len(self.viewed)
        pts = len(self.selected)
        var = len(self.variables)
        return "{}\n" \
               "INSTANTS : {} instant(s) sélectionnés parmis {} courbes " \
               "observées et {} variables.".format(Opset.__repr__(self), pts, ops, var)

    
    def clear_selection(self):
        """ Réinitialise la liste des sélections et observations.
        
            Il est utile d'appeler cette méthode avant un réapprentissage
            si vous ne voulez pas garder toute la sélection précédente.
            Par exemple, il est possible de modifier les instants sélectionnés
            graphiquement, mais pas de supprimer des variables ou des signaux
            vus précédemment.
        """
        self.viewed.clear()
        self.selected.clear()
        self.variables.clear()
        self.computed.clear()
        
    
    # ================== Apprentissage ====================================
    def make_indicators(self, idfilename=None, progress_bar=None, message_label=None):
        """ Construction de l'Opset des indicateurs.
        
            Pour toutes les opérations pour lesquelles l'expert à identifié
            un instant d'intérêt, et toutes les variables observées, on va
            créer un jeu d'indicateurs en fonction des paramètres de `features`.
            
            Cet Opset est en fait un Selector pour lequel le dictionnaire
            `computed` est renseigné, ce qui permet de visualiser sur les 
            indicateurs les points à identifier.
            
            Cette méthode est appelée automatiquement par `fit()`, mais on
            peut l'utiliser directement pour observer les indicateurs produit.
            Cela permet de mieux calibrer les paramètres correspondant à la
            descriptions des indicateurs (`feature_params`).
        """

        if not self.selected:
            raise OpsetError(self.storename,"Nothing to learn !")
            
        # Nom du fichier de sauvegarde
        if idfilename is None:
            i = self.storename.rfind('.')
            idfilename = self.storename[:i] + '_I' + self.storename[i:]
        
        # Liste des variables
        colnames = self.variables
        # Liste des observations et instants selectionnés
        obs = self.selected.keys()
        ind = self.selected.values()
        
        # Mémorisation des données utiles dans une liste data.
        data = [] # Une liste de matrices de signaux.
        L = []    # Les longueurs de chaque signal.
        for df in self[obs]:
            L = L +[len(df)]
            data.append(df[colnames])
        Q = np.array(list(ind))/np.array(L)
        Qmin = Q.min()
        Qmax = Q.max()
        
        
        # Demi largeur minimale du filtre.
        if self.feature_params['range_width'] is None:
            L0 = max(10,int(np.floor(np.min(L)/100)))
            self.feature_params['range_width'] = range(L0,L0*10+1,L0)
        
        # Ordre de dérivation.
        max_order = self.feature_params['max_order']
        deg_poly = max(2,max_order)

        # Barre de progression.
        range_width = self.feature_params['range_width']
        range_sigma = self.feature_params['range_sigma']
        if progress_bar:
            add_max = len(data)*len(colnames)*len(range_width)*max_order*2
            progress_bar.max += add_max
            
        # Calcul des epsilons.
        if message_label:
            message_label.value = "Estimating local variances ..."
        epsilon = dict()
        for l in range_width:
            w = 2*l+1 # La largeur doit être impaire pour SG.
            R = np.ndarray((max_order,len(colnames)))
            for d in range(max_order):
                for i,colname in enumerate(colnames):
                    r = []
                    k = 0
                    for df in data:
                        if progress_bar:
                            progress_bar.value += 1
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
            
        # Calcul des indicateurs.
        if message_label:
            message_label.value = "Computing indicators ..."
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
                idcodes = [('LEN',0,0,0,0.0), ('REV',0,0,0,0.0),
                           ('PERCENT',0,0,0,0.0)]
            for i,colname in enumerate(colnames):
                a = df[colname].values
                C = np.vstack((C,a))
                if first:
                    idnames = idnames + [colname]
                    idcodes = idcodes + [(colname,0,0,0,0.0)]
                name,unit = nameunit(colname)
                for l in range_width:
                    w = 2*l + 1
                    for d in range(max_order):
                        if progress_bar:
                            progress_bar.value += 1
                        eps = epsilon[w][d,i]
                        for s in range_sigma:
                            for e in [1, -1]:
                                c = indicator(a,w,d+1,e*s*eps,deg_poly)
                                if Qmin<0.65:
                                    C = np.vstack((C,c))
                                if Qmax>0.35:
                                    c = c[-1]-c
                                    C = np.vstack((C,c))
                                if first: # len(k)>0 and k not in K:
                                    code = '{}o{}'.format(l, d+1)
                                    if e>0:
                                        code = code + 'u{}'.format(s)
                                    else:
                                        code = code + 'd{}'.format(s)
                                    if Qmin<0.65:
                                        idnames = idnames +\
                                             ["{}[+w{}]".format(name,code),
                                              "{}[-w{}]".format(name,code)]
                                    if Qmax>0.35:
                                        idcodes = idcodes +\
                                            [(colname,  l, d, e*s, eps),
                                             (colname, -l, d, e*s, eps)]
            first = False
            dfi = pd.DataFrame(C.T, index=df.index, columns=idnames)
            self.idcodes = idcodes # On sauvegarde les indicateurs.
            dsi.put(dfi) # Le nm de l'opération est dans l'index.
        
        # Pour visualiser les données sur les signaux 
        # on conserve les instants sélectionnés.
        dsi.computed = dict(enumerate(ind))
        dsi.rewind()
        dsi.idcodes = self.idcodes
        self._dsi = dsi
        return dsi
    
    
    def fit(self, progress_bar=None, message_label=None):
        """ Apprend un arbre de décision basé sur les indicateurs.

            Cette fonction crée les indicateurs par l'appel de 
            `make_indicators()` si `_dsi` est vide ou si les variables
            observées ne sont plus les mêmes.
            
            Plusieurs arbres de décision sont d'abord testés pour extraire
            une liste réduite d'indicateurs pertinets. Ensuite on recrée un
            arbre définitif à partir de cette sous-sélection.
        """
        
        # #################################################################
        # ----------- Fabrication des données d'apprentissage -------------
        def find_best_parameters(p,split,cols=None):
            X = []
            Y = []
            i0 = 0
            for df in dsi:
                n = len(df)
                ind = dsi.computed[dsi.sigpos]
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
        # ------------- Fin du code d'optimisation ----------------
        
        # Barre de progression.
        rn = self.learn_params['retry_number']
        dsi = self._dsi
        if progress_bar:
            progress_bar.max = rn+3
            progress_bar.value = 1
            
        # On vérifie s'il faut recréer les indicateurs.
        if dsi:
            colnames = {c[0] for c in dsi.idcodes[3:]}
            if colnames != self.variables or \
                           list(dsi.computed.values()) != list(self.selected.values()):
                dsi = None
        if dsi is None:
            if message_label:
                message_label.value = "Rebuilding indicators ..."
            dsi = self.make_indicators(progress_bar=progress_bar,
                                       message_label=message_label)
                
    
        if message_label:
            message_label.value = "Searching features ..."
        fi = np.zeros(len(dsi.df.columns))
        for k in range(rn):
            clf =  find_best_parameters(self.learn_params['samples_percent'],
                                        self.learn_params['min_samples_split'])
            fi += clf.feature_importances_
            if progress_bar:
                progress_bar.value += 1
                
        seuil = np.percentile(fi,self.learn_params['retry_percentile'])
        keepcols = np.argwhere(fi>seuil).ravel().tolist()
        
        # Apprentissage final.
        if message_label:
            message_label.value = "Searching rules ..."
        print("First keeping {} indicators over {}"\
              .format(len(keepcols),len(dsi.idcodes)))
        p1 = self.learn_params['samples_percent']*self.learn_params['retry_number'] ;
        clf = find_best_parameters(min(0.5,p1),
                                   self.learn_params['min_samples_split'],
                                   keepcols)
        fi = clf.feature_importances_
        if progress_bar:
                progress_bar.value += 1
                
        while np.sum(fi==0)>0:
            newcols = np.array(keepcols)[fi>0]
            keepcols = newcols.tolist()
            clf = find_best_parameters(min(0.5,p1),
                                   self.learn_params['min_samples_split'],
                                   keepcols)
            fi = clf.feature_importances_
                       
        if progress_bar:
                progress_bar.value += 1
        print("Then  keeping {} indicators over {}"\
              .format(len(keepcols),len(dsi.idcodes)))
        self.idcodes = [dsi.idcodes[i] for i in keepcols]
        self._clf = clf
        self.computed = dict()
        
        return self
    
    
    def describe(self):
        """ Affiche l'arbre de décision final."""
        if self._clf is None:
            print("Nothing yet!")
            return
        
        codes = pd.DataFrame(data=self.idcodes, 
                             columns=['Name', 'Filter', 'Order', 'Sigma', 'Std'])
        codes.index.name = "Feature"
        print(codes)
        r = tree.export_text(self._clf)
        print(r)
        
    
    def belief(self, arg=None):
        """ Calcul d'un indicateur de présomption de détection.
        
            Cette fonction recherche un point pour lequel les estimations
            sont négatives à gauche et positives à droite.
            On utilise la valeur max d'une dérivée.
            
            Le paramètre `predict_params.filter_width` donne la demi-largeur
            du lissage pour le calcul de la dérivée.
        """
        
        if arg is None:
            df = self.df
            extern = False
        elif isinstance(arg,pd.DataFrame):
            df = arg
            extern = True
        elif isinstance(arg,int):
            df = self.rewind(arg).df
            extern = False
        else:
            raise OpsetError(self.storename, "Bad argument.")
            
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
                    c = df[colname].values
            else:
                a = df[colname].values
                w = 2*np.abs(l) + 1
                deg_poly = max(self.feature_params['max_order'],d)
                c = indicator(a,w,d+1,es*eps,deg_poly)
            if l<0:
                c = c[-1]-c
            C = np.vstack((C,c))
            
        ip = clf.predict(C.T) # Eventuellement pondérer.
        filter_width = self.predict_params['filter_width']
        p = signal.savgol_filter(ip,
                                 window_length=2*filter_width+1,
                                 polyorder=2,
                                 deriv=1)
        
        p = np.maximum(p,0)
        Z = p.sum()
        if Z == 0.0:
            Z=1.0
        p /= Z
        
        if not extern:
            mx = np.argmax(p)
            self.computed[self.sigpos] = mx
        
        return p
    
    
    def predict(self, arg=None):
        """ Renvoie la liste des instants prédits.
        
            :return: Un dictionnaire avec les instants calculés.
        """
        
        if arg is None:
            if len(self.computed) == len(self.records):
                return self.computed
            
            sigpos = self.sigpos
            for df in self:
                self.belief()
            self.rewind(sigpos)
            return self.computed
            
        else:
            if isinstance(arg,Opset):
                ds = arg
            else:
                ds = Opset(arg)
            r = dict()
            sigpos = ds.sigpos
            for i,df in enumerate(ds):
                p = self.belief(df)
                mx = np.argmax(p)
                r[i] = mx
            ds.rewind(sigpos)
            return r
    
    
    def left(self,filename=None):
        """ Extrait la partie amont d'un signal."""
    
        if self._clf is None:
            raise OpsetError(self.storename,"Need learning before.")
            
        sigpos = self.sigpos
        
        # Nom du fichier de sauvegarde
        if filename is None:
            i = self.storename.rfind('.')
            filename = self.storename[:i] + 'L' + self.storename[i:]

        dsl = Selector(filename).clean()
        for df in self:
            if not self.sigpos in self.computed:
                self.belief()
            i = self.computed[self.sigpos]
            dsl.put(df.iloc[:i])
        
        self.rewind(sigpos)
        return dsl.rewind()

    
    def right(self,filename=None):
        """ Extrait la partie aval d'un signal."""
    
        if self._clf is None:
            raise OpsetError(self.storename,"Need learning before.")
    
        sigpos = self.sigpos
        
        # Nom du fichier de sauvegarde
        if filename is None:
            i = self.storename.rfind('.')
            filename = self.storename[:i] + 'R' + self.storename[i:]

        dsr = Selector(filename).clean()
        for df in self:
            if not self.sigpos in self.computed:
                self.belief()
            i = self.computed[self.sigpos]
            dsr.put(df.iloc[i:])
        
        self.rewind(sigpos)
        return dsr.rewind()
    
    
    def between(self,left,right,filename=None):
        """ Découpe el signal entre l'instant et une autre borne.
        
            Les paramètres `left` et `right` doivent être des dictionnaire de
            sélection ou des Selectors.
        """

        if isinstance(left,Selector):
            left = left.predict()
        if isinstance(right,Selector):
            right = right.predict()
                   
        # Nom du fichier de sauvegarde
        if filename is None:
            i = self.storename.rfind('.')
            filename = self.storename[:i] + 'B' + self.storename[i:]

        sigpos = self.sigpos
        
        dsb = Opset(filename).clean()
        for df in self:
            i = left[self.sigpos]
            j = right[self.sigpos]
            dsb.put(df.iloc[i:j])
        
        self.rewind(sigpos)
        return dsb.rewind()
    
    
    def all_scores(self):
        """ Renvoie les écarts entre détection et label."""
        
        if self._clf is None:
            return []
        
        scores = dict()
        for i in self.selected:
            t0 = self.selected[i]
            if not i in self.computed:
                self.belief(i)
            t1 = self.computed[i]
            scores[i] = t1-t0

        return scores
                
        
    def score(self):
        """ Renvoie l'écart maximal absolu de détection."""
        
        if self._clf is None:
            return np.nan
        
        scores = self.all_scores()
        
        return np.max(np.abs(list(scores.values())))
    
    
    def load(self,filename):
        """ Recharge un nouveau fichier à analyser."""
        ds = Selector(filename,self.phase)
        ds.idcodes = self.idcodes
        ds._clf = self._clf
        
        return ds
        
    # ====================== Affichages ===========================                              
    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Création de l'interface graphique.
        
            On rajoute à l'interface graphique de l'Opset une fonction de
            sélection d'instant sur la courbe affichée (il faut gérer le
            cas de la surimpression d'une phase) et une ligne contenant :
                - un slider pour la taille du filtre de prédiction ;
                - un bouton pour lancer ou relancer l'apprentissage.
                
            Le boutant d'apprentissage est rouge tant qu'il n'y a aucune
            donnée labellisée, il passe au bleu quand on peut lancer un
            premier apprentissage (méthode `fit()`), puis au vert quand
            l'apprentissage s'est bien passé et que l'on peut recommencer.
        """

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
            self.predict_params['filter_width'] = width
            f.update_traces(x=self.df.index, y=self.belief(), row=2)
            
            self.viewed.add(sigpos)
            f.layout.shapes = []
            if self.sigpos in self.selected:
                i = self.selected[self.sigpos]
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
                i = self.computed[self.sigpos]
                x0 = self.df.index[i]
                y0 = min(self.df[colname])
                y1 = max(self.df[colname])
                f.add_shape(type='line',
                                  x0=x0, y0=y0, x1=x0, y1=y1,
                                  line= {'color': 'rgb(96, 50, 171)',
                                        'width': 2,
                                        'dash': 'dot'},
                                  row=1, col=1)
        # ---- End: Callback Interactive ----
        
        # =================================================================
        # --------- Liste de sélection des variables à prédire ------------
        wlmv = widgets.SelectMultiple(options = self.df.columns,
                                      value = tuple(self.variables),
                                      description = 'To learn',
                                      rows=8,
                                      disabled = False)

        e['variable_selection'] = wlmv
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
            self.variables = set(wlmv.value)
            
            wml.value = ""
            b.button_style = 'warning'
            b.description = 'Learning...'
            if not self.selected:
                b.description = 'No Target'
                b.button_style = 'danger'
            else:
                self.fit(progress_bar=wp, message_label=wml)
                update_plot(self.colname,self.sigpos,wf.value)
                b.description = 'Relearn'
                b.button_style = 'success'
                wml.value = ""
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
                self.variables.add(self.colname)
                wlmv.value = tuple(self.variables)
                self.selected[self.sigpos] = i1
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
        wf = widgets.IntSlider(value=self.predict_params['filter_width'], 
                               min=10, max=500, step=10,
                               orientation='horizontal',
                               description='Filter',
                               continuous_update=False,
                               layout=widgets.Layout(width='500px'))
           
            
        # Afficher la barre si un signal est présent.
        update_plot(self.colname,self.sigpos,wf.value)

        # On remplace la fonction d'update (que l'on avait d'abord copiée).
        e['update_function'] = update_plot 
        e['filter_slider'] = wf
        e['learn_button'] = wbl
        return e
    
    
    def param(self):
        """ Interface de paramétrage.
        
            Cette fonction affiche dans une cellule de dialogue les paramètres
            courants et offre la possibilité de les modifier graphiquement.
        """
        
        def update_parameters(width, sigma, max_order,
                              retry, percentile, sample, split):
            
            self.feature_params['range_width'] = eval(width)
            self.feature_params['range_sigma'] = eval(sigma)
            self.feature_params['max_order'] = max_order
            
            self.learn_params['retry_number'] = retry 
            self.learn_params['retry_percentile'] = percentile
            self.learn_params['samples_percent'] = sample 
            self.learn_params['min_samples_split'] = split
            
          
        # =================================================================
        # Widgets pour les indicateurs.
        wfrw = widgets.Textarea(description = "Range", 
                               value = repr(self.feature_params['range_width']),
                               placeholder = '____________________')
        wfrs = widgets.Textarea(description = "Sigma", 
                               value = repr(self.feature_params['range_sigma']),
                               placeholder = '____________________')
        wfmo = widgets.IntText(description = "Max Order", 
                               value = self.feature_params['max_order'])

        fbox = widgets.VBox([widgets.Label("Feature parameters (.feature_params):"),
                             widgets.HBox([wfrw, wfrs]),
                             widgets.HBox([wfmo])])
        
        
        # =================================================================
        # Widgets pour l'apprentissage.
        wtrn = widgets.IntText(description = "Retry", 
                               value = self.learn_params['retry_number'])
        wtrp = widgets.IntText(description = "Percentile", 
                               value = self.learn_params['retry_percentile'])
        wtsp = widgets.FloatText(description = "Sample", 
                                 value = self.learn_params['samples_percent'])
        wtss = widgets.FloatText(description = "Split", 
                                 value = self.learn_params['min_samples_split'])
        
        lbox = widgets.VBox([widgets.Label(value="Learning parameters (.learn_params):"),
                             widgets.HBox([wtrn, wtrp]),
                             widgets.HBox([wtsp, wtss])])
        
        out = widgets.interactive(update_parameters,
                                  width=wfrw, sigma=wfrs, max_order=wfmo,
                                  retry=wtrn, percentile=wtrp,
                                  sample=wtsp, split=wtss)

        return widgets.VBox([fbox, lbox])
        
    
    def plot(self,phase=None,pos=None,name=None):
        """ On ajoute à l'affichage de l'Opset une sélection d'instants."""
        f = make_subplots(rows=2, cols=1,
                          shared_xaxes=True,
                          vertical_spacing=0.03,
                          specs=[[{"type": "scatter"}],
                                [{"type": "scatter"}]])
        f = go.FigureWidget(f)
        e = self.make_figure(f,phase,pos,name)
        out = widgets.interactive(e['update_function'], 
                                  colname=e['variable_dropdown'], 
                                  sigpos=e['signal_slider'],
                                  width=e['filter_slider'])
        
        boxes = widgets.VBox(
            [widgets.HBox([e['variable_dropdown'], 
                          e['previous_button'], e['next_button']]),
             widgets.HBox([f, e['signal_slider']]),
             widgets.HBox([e['filter_slider']])
            ])
        
        params = self.param()
        
        learn = widgets.VBox([e['progress_bar'], 
                              widgets.HBox([e['variable_selection'], 
                                            widgets.VBox([e['message_label'],
                                                          e['learn_button']])])])
                              
        tabs = widgets.Tab(children = [boxes, params, learn])
        tabs.set_title(0, "Plot")
        tabs.set_title(1, "Param")
        tabs.set_title(2, "Learn")
        
        return tabs
    
    
