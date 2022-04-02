# -*- coding: utf-8 -*-
"""
BANALISE - Transformation des données pour un export.

Created on Mon May 14 17:37:51 2018

@author: Jérôme Lacaille
"""

import numpy as np
import pandas as pd
from tabata.opset import nameunit
from datetime import datetime,timedelta
from itertools import tee
import os

#%% Constantes.
DELAY = timedelta(days=11*365+65,hours=2,minutes=12) 


#%% Banalisation.
def banalise(dfiterator,storename,comment=None,nbmax=None,
             delay=DELAY,iqr=0.25,nanvalues=-999.0,
             aliasfile=None):
    """
    ETDEX.BANALISE - Crée un HDFStore de données banalisées suivant la
    procédure suivante :
        * (1) Supprime les colonnes toujours NAN ou toujours constantes 
                (remplace aussi les valeurs -999 par des NAN).
        * (2) Recrée des unités propres.
        * (3) Calcule les maximums et minimums des interquantiles des 
              données par unité.
        * (4) Par unité le facteur multiplicatif appliqué est la partie entière
              du max des interquantiles divisée par l'ordre de grandeur en 
              base 10.
        
    Crée aussi un fichier contenant la date de banalisation et les opérations
    effectuées avec un facteur multiplicatif par unité et un décalage 
    temporel global (delay).
    """
    
    #%% Initialisation, préparation des fichiers de sortie.
    datenow = str(datetime.now()).split(' ')[0]
    i = storename.find('.')
    # if i >= 0:
    #     storename = storename[:i]+"("+datenow+")"+storename[i:]
    # else:
    #     storename = storename+"("+datenow+").h5"
    i = storename.find('.')
    logname = storename[:i]+".log"
    
    with open(logname,'w') as log:
        if comment is not None:
            print(comment, file=log)
            print(file=log)
        print("Date de la banalisation      :", datenow, file=log)
        print("Fichier de sauvegarde hdf5   :", storename, file=log)
        print("Fichier de log               :", logname, file=log)
        print(file=log)

    #%% Gestion des valeurs manquantes.
    # Il faut repasser en liste.
    if type(nanvalues) is not list:
        nanvalues = [nanvalues]

    #%% Multiple parcours de l'iterateur.
    iter1,iter2 = tee(dfiterator,2)

    #%% Récupération des alias.
    # Un fichier d'alias permet de convertir les noms de variables.
    alias = dict()
    if aliasfile is not None:
        if os.path.isfile(aliasfile):
            with open(aliasfile,'r') as f:
                for line in f:
                    if len(line) > 0 and line[0] != '#':
                        r = line.split()
                        if len(r) == 2:
                            alias[r[0]]=r[1]
    if len(set(alias.values())) < len(alias):
        ralias = dict()
        for key, val in alias.items():
            if val in ralias:
                ralias[val].append(key)
            else:
                ralias[val] = [key]
        for val in ralias:
            if len(ralias[val]) > 1:
                print("{:12} {}".format(val, ralias[val]))
        raise ValueError("Attention ! Au moins deux labels différents ont le même alias.")

    #%% Parcours des observations.
    # On récupère la liste des labels et les unités associées.
    # On calcules les min, max et interquantiles de chaque observation.
    of = dict()
    ct = 0
    for df in iter1:
        if nbmax is not None and ct == nbmax:
            break
        ct += 1
        nu = [nameunit(col) for col in df.columns]

        df = df.replace(nanvalues+[np.inf, -np.inf],np.nan)
        dq = df.quantile([0, iqr,1-iqr, 1],axis=0)
        
        for k,(n,u) in enumerate(nu):
            q = list(dq.iloc[:,k])
            if id in of:
                m0,m1,m2,m3,us = of[n]
                m0 = min(m0,q[0])
                m1 = min(m1,q[1])
                m2 = max(m2,q[2])
                m3 = max(m3,q[3])
                if u != '-' and u not in us:
                    us.append(u)
                of[n] = (m0,m1,m2,m3,us)
            else:
                if u == '-':
                    of[n] = (*q,[])
                else:
                    of[n] = (*q,[u])
    
    with open(logname,'a') as log:
        print("Nombre d'enregistrements     :", ct, file=log)
        print("Nombre de labels             :",len(of), file=log)
    
    #%% Suppression des labels et choix de l'unité.
    # Certains labels sont constants sur tous les fichiers : on les supprime. 
    # On extrait l'unité unique correspondant à chaque label.
    of2 = dict()
    for label,(m0,m1,m2,m3,us) in of.items():
        if m3 > m0:  # On n'observe que les variables non constantes.
            if len(us) > 0:  # Gestion de l'unité.
                u = us[0]
                r = u.split('_')  # On supprime à droite du _.
                u = r[0]
                r = u.split(' ')  # On supprime à droite du blanc.
                u = r[0]
                if u == '°' or u == '°C':
                    u = 'C'
                elif u == '°K':
                    u = 'K'
            else:
                u = '-'
            of2[label] = (m1,m2,u)  # On ne conserve que les interquantiles.

    with open(logname,'a') as log:
        print("Nb de labels non constants   :",len(of), file=log)
    
    #%% Traitement par unité.
    # L'idée est d'avoir un calcul identique pour tous les labes de même unité.
    ofunit = dict()
    for label,(m1,m2,u) in of2.items():
        if u in ofunit:
            mu1,mu2 = ofunit[u]
            mu1 = min(mu1,m1)
            mu2 = max(mu2,m2)
            ofunit[u] = (mu1,mu2)
        else:
            ofunit[u] = (m1,m2)

    with open(logname,'a') as log:
        print("Nombre d'unités distinctes   :",len(ofunit), file=log)
    
    #%% Calcul du facteur de correction.
    # On décide de concerver la signification du zéro. 
    # Ausi on se contente de diviser par la partie entière supérieure du max 
    # des valeurs absolues des inter-quantiles. 
    # Pour garder l'ordre de grandeur, on divise par l'ordre de grandeur le 
    # plus proche.
    # Si le rapport est égal à 1, on rajoute une homothétie de 1.163.
    ofunit2 = dict()
    for u,(m1,m2) in ofunit.items():
        factor = np.ceil(max(abs(m1),abs(m2)))
        if factor == 0:
            factor = 1
        else:
            factor /= 10**np.round(np.log10(factor))
            if factor == 1:
                factor *= 1.163
        factor = np.round(100*factor)/100
        ofunit2[u] = factor
    
    # Affichage des facteurs de correction.
    with open(logname,'a') as log:
        print(file=log)
        print("Facteurs de correction par label", file=log)
        for u,factor in ofunit2.items():
            print("{:>28s} : {:.2f}".format(u,factor), file=log)
    
    #%% Opération de normalisation.
    # Les labels à conserver sont dans of2, les unités avec le facteur 
    # multiplicatif dans ofunit2. Si jamais un enregistrement n'a plus de label
    # à conserver on ne l'enregistre pas. Of2 ne gère que les labels nan ou
    # constants sur l'ensemble des enregistrements. On peut encore enlever les
    # colonnes de nan par fichier car elles n'apportent rien. Par contre on
    # garde les colonnes constantes qui restent car elles peuvent avoir une 
    # autre valeur dans un autre enregistrement.
    #
    # Pour l'index on décide d'enlever 
    #     11 ans et 2 mois et 3 jours, 2 heures et 12 minutes. 
    #
    # Les noms des essais sont appelés record_001, record_002 ... 
    # Les numéros d'enregistrement supprimés sont comptabilisés, il peut donc 
    # y avoir un trou dans le fichier HDF5.
    #
    # On garde bien un décalage unique sur tous les fichiers pour ne pas 
    # intervenir sur la superposition des enregistrements.
    nc = int(np.ceil(np.log10(ct)))
    rname = "record_{:0"+str(nc)+"d}"
    n = 0
    unused_records = 0
    suppressed_measures = 0
    new_aliases = 0

    with open(logname, 'a') as log:
        print("Enregistrements :", file=log)
    with pd.HDFStore(storename, mode='w') as bana:
        ct = 0
        for df in iter2:
            if nbmax is not None and ct == nbmax:
                break
            ct += 1
            nu = [nameunit(col) for col in df.columns]
            names = [it[0] for it in nu]

            df = df.replace(nanvalues,np.nan)
            df.index -= delay
            keep = [False]*len(df.columns)
            for i,(name,unit) in enumerate(nu):
                if unit in ofunit:
                    factor = ofunit2[unit]
                    if name in of2.keys() and not df.iloc[:,i].isna().values.all():
                        keep[i] = True
                        df.iloc[:,i] *= factor
            if any(keep):
                ds = df[df.columns[keep]]
                names = [names[i] for i,k in enumerate(keep) if k]
                units = [of2[name][2] for name in names]  # L'unité conservée.
                # On change les noms pr les alias.
                newnames = []
                for name in names:
                    if name in alias:
                        newnames.append(alias[name])
                    else:
                        newnames.append(name)
                        alias[name] = name
                        new_aliases += 1
                ds.columns = [name+"["+unit+"]" for (name,unit) in zip(
                    newnames,units)]
                recordname = rname.format(n)
                with open(logname, 'a') as log:
                    print("{:12}\t".format(recordname),ds.index.name,
                          file=log)
                ds.index.name = recordname
                bana.put(recordname,ds)
            else:
                unused_records += 1
            suppressed_measures += len(keep)-sum(keep)
            n += 1
    
    if new_aliases>1 and aliasfile is not None:
        with open(aliasfile,'w') as f:
            for name in alias:
                print("{:12}\t{:12}".format(name,alias[name]), file=f)
    
    with open(logname,'a') as log:
        print(file=log)
        print("Delay temporel retranché     :", delay, file=log)
        print("Enregistrements non utilisés :", unused_records, file=log)
        print("Mesures inutiles supprimées  :", suppressed_measures, file=log)
        print("Nouveaux alias à revoir      :", new_aliases, file=log)

        if aliasfile is not None:
            print("Alias :", file=log)
            for name in alias:
                print("{:12}\t{:12}".format(name, alias[name]), file=log)


#%% Tests.
if __name__ == "__main__":
    from pysamanta import etdex

    dirname = os.environ['HOME']+"wrk/bancs/data/sam146/"
    expname = os.environ['HOME']+"wrk/bancs/extract/"
    bananame = "banal.h5"

    banafile = os.path.join(expname,bananame)
    aliasfile = os.path.join(expname,"alias.txt")

    iterator = etdex.dataiterator(dirname, nbmax=5)
    banalise(iterator,banafile,
             aliasfile = aliasfile,
             comment="SAM146 - 2007",
             nbmax=5)
