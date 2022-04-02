# -*- coding: utf-8 -*-
"""
ETDEX - Gestion des fichiers ETDEX.

Created on Wed Apr 11 18:00:46 2018

@author: s068990
"""

import os
import struct
import numpy as np
import pandas as pd
from datetime import datetime
from lxml import etree
# Doc de lxml : http://infohost.nmt.edu/tcc/help/pubs/pylxml/web/index.html

# ----------------------------------------------------------------------------
class EtdexError(ValueError):
    def __init__(self,filename,message):
        self.filename = filename
        self.message = message
        
    def __str__(self):
        return """Etdex({self.filename})
    {self.message}""".format(self=self)


# ----------------------------------------------------------------------------
class Etdex:
    """
    ETDEX - Gestion du fichier Etdex
    """  

    # ------------------------------------------------------------------------
    def __init__(self,filename):
        """
        ETDEX.__INIT_ - Lecture des caractéristiques d'un fichier ETDEX.
        """
        
        def get_date(dt):
            """
            GET_DATE - récupère la date codée dans un élément XML
            """
            d = dt.find("DATE")
            day = d.get("DAY")
            year = d.get("YEAR")
            month = d.get("MONTH")
            t = dt.find("TIME")
            mn = t.get("MIN")
            sec = t.get("SEC")
            hr = t.get("HOUR")
            return datetime(int(year), int(month), int(day),
                            int(hr), int(mn), int(sec))

        # On ouvre le fichier XML.
        self.filename = filename
        tree = etree.parse(filename)
        
        # Récupération du banc.
        u = tree.xpath("/READINGS/INFORMATIONS/CELL")
        if len(u) > 0:
            self.banc = u[0].get("NAME")
        
        # Le moteur et son montage.
        u = tree.xpath("/READINGS/INFORMATIONS/ENGINE")
        if len(u) > 0:
            self.moteur = u[0].get("FAMILY")
        
        u = tree.xpath("/READINGS/INFORMATIONS/ENGINE/IDENT")
        if len(u) > 0:
            self.numero = u[0].get("NUM_MOTEUR")
            self.montage = u[0].get("NUM_MONTAGE")
    
        # La date de l'essai peut être différente du début des enregistrements.
        u = tree.xpath("/READINGS/INFORMATIONS/TEST/TEST_TIME/START")
        if len(u) > 0:
            self.date = get_date(u[0])
        
        # Début et fi du signal
        u = tree.xpath("/READINGS/TRANSIENT_CHARACTERISTICS/" +
                       "TRANSIENT_ACQUISITION/ACQUISITION_TIME")
        self.start = get_date(u[0].find('START'))
        self.end = get_date(u[0].find('END'))
        duration = u[0].find('DURATION')
        if duration is None:
            self.duree = (self.end-self.start).total_seconds()
            if self.duree == 0:
                self.duree = 0.001  # On  vaisemblablement que 1 point de
                                    # mesure.
        else:
            self.duree = (float(duration.get('NB_SEC'))*1000 + 
                          float(duration.get('NB_MSEC')))/1000
        
        # Fréquence unique
        u = tree.xpath("/READINGS/TRANSIENT_CHARACTERISTICS")
        span = 0.0
        freq = 0.0
        if len(u) > 0:
            self.transient = u[0].get('TRANSIENT_TYPE')
            sfreq = u[0].get("FREQUENCY")
            if sfreq is not None:
                freq = float(sfreq)
                span = 1/freq
            
        # Paramètres.
        u = tree.xpath("/READINGS/TRANSIENT_CHARACTERISTICS/" +
                       "TRANSIENT_ACQUISITION/PARAMETER")
        
        # On teste si le format des données est bien unique est IEEE_R4.
        sfmts = {p.find('MEASUREMENT_TR').get('FORMAT_VALUE') for p in u}
        if len(sfmts) != 1 or 'IEEE_R4' not in sfmts:
            raise EtdexError(self.filename,
                             "Only 'IEEE_R4' format is managed yet.")
        
        # Nombre de mesures.
        sz = [int(p.find('MEASUREMENT_TR').get('NB_VALUES')) for p in u]
        ssz = list(set(sz))
        if len(ssz) != 1:
            raise EtdexError(self.filename,
                             "Manage only files of same length yet.")
        nb = ssz[0]
        
        # Calcule la fréquence d'acquisition si non déclarée.
        if span == 0.0:
            if self.duree == 0:
                raise EtdexError(self.filename,"Essai de durée nulle.")
            freq = nb/self.duree
            if freq == 0:
                raise EtdexError(self.filename,"Fréquence nulle.")
            span = 1/freq

        # Création de l'index
        if freq < 1:
            self.freq = round(freq,2)
            self.sfreq = str(round(freq,2))+" Hz"
            self.span = str(round(span))+"S"
        else:
            self.freq = round(freq)
            self.sfreq = str(self.freq)+" Hz"
            self.span = str(round(span*1000))+"L"
        self.index = pd.date_range(self.start,periods=nb,freq=self.span)
        self.index.name = os.path.basename(self.filename)[:-4]

        # Récupération des noms des mesures.
        self.names = [p.get('NAME') for p in u]
        self.units = ['-' if p.get('UNIT') is None else p.get('UNIT') for p in u]
        self.columns = [n+"["+u+"]" for (n,u) in zip(self.names,self.units)]
    
        # Fichier binaire.
        u = tree.xpath("/READINGS/TRANSIENT_CHARACTERISTICS/" +
                       "TRANSIENT_ACQUISITION/FILE")
        if u[0].get('TYPE') != 'BINARY':
            raise EtdexError(self.filename,
                             "Don't know how to read non binary data.")
        fname = u[0].get('NAME')
        self.datfile = os.path.join(os.path.dirname(filename),fname)
        self.encodage = "Unknown"

        # Récupération de l'état du binaire.
        self.check_encoding()

    # ------------------------------------------------------------------------
    def __repr__(self):
        """
        La représentation d'un appel à l'objet Etdex.
        """
        return "Etdex({})".format(self.filename)
    
    def __str__(self):
        """
        La description du contenu n'un Etdex.
        """
        return """Etdex({self.filename}) - {self.encodage}
    Banc {self.banc} test le {self.date} de {self.duree} secondes
    {self.moteur} Moteur {self.numero} Montage {self.montage}
    {nbm} mesures, fréquence {self.sfreq}, {nbv} variables.""".format(
            self=self,nbm=len(self.index),nbv=len(self.names))

    # ------------------------------------------------------------------------
    def check_encoding(self):
        """
        ETDEX.CHECK_ENCODING - Récupère le codage du fichier binaire.
        """
        datfile = self.datfile
        f = open(datfile,'rb')

        skip = struct.unpack('>B',f.read(1))[0]
        if skip != 11:
            raise EtdexError(self,
                             "Taille du cartouche {} différente de 11.".format(skip))
        check_float = struct.unpack('>f',f.read(4))[0]
        check_int = struct.unpack('>I',f.read(4))[0]
        check_short = struct.unpack('>H',f.read(2))[0]
        f.close()
      
        # On teste le cas big-endian d'abord.
        self.encodage = "big-endian"
        if (abs(check_float - 0.2) > 1e-6) or (check_int != 0xFAFF0F) or (check_short != 0xFA):
            self.encodage = "little-endian"
            skip = struct.unpack('<B',f.read(1))[0]
            if skip != 11:
                raise EtdexError(self,
                                 "Taille du cartouche {} différente de 11.".format(
                                     skip))
            check_float = struct.unpack('<f',f.read(4))[0]
            check_int = struct.unpack('<I',f.read(4))[0]
            check_short = struct.unpack('<H',f.read(2))[0]
            if not ((abs(check_float - 0.2) > 1e-6) or (check_int != 0xFAFF0F) or (check_short != 0xFA)):
                raise EtdexError(self.datfile,
                                 "Mauvais encodage du fichier binaire.")
                
    # ------------------------------------------------------------------------
    def data(self):
        """
        ETDEX.VALUES - Récupère les données du fichier binaire.

        L'index contient le nom du fichier d'origine.
        """
        
        # Récupération des taille à lire.
        n = len(self.index)    # Nombre de lignes.
        p = len(self.columns)  # Nombre de colonnes.
        
        # Lecture du fichier binaire.
        f = open(self.datfile,'rb')
        f.seek(11)
        buffer = f.read()
        f.close()
        
        # Traitement du buffer
        v = struct.unpack(">{}f".format(n*p),buffer)
        a = np.array(v).reshape(p,n)
        df = pd.DataFrame(a.T, index=self.index, columns=self.columns)
        # df.name = os.path.basename(self.filename)[:-4]
        # Le nom du dataFrame n'est pas persistent.

        return df


#%% Nombre de fichiers ETDEX transitoires d'un répertoire.
def nbtransient(dirname):
    """
    ETDEX.NBTRANSIENT - Renvoie le nombre de fichiers ETDEX transitoires.
    """
    etdexlist = [filename for 
                 filename in os.listdir(dirname) if filename[-4:] == '.dat']
    return len(etdexlist)
    

#%% Un itérateur sur les fichiers ETDEX d'un répertoire.
def iterator(dirname,nbmax=None):
    """
    EDTEX.ITERATOR - Renvoie un itérateur sur tous les fichiers ETDEX
    transitoire d'un dossier.
    
    Les éléments retournés sont des objets de la classe Etdex. En
    particulier :
        - Etdex.data() renvoie un DataFrame
        - Etdex.freq donne la fréquence.
        - Etdex.banc
        - Etdex.moteur 
        - Etdex.numero
        - Etdex.montage
        ...
    """
    etdexlist = [filename[:-4]+'.xml' for 
                 filename in os.listdir(dirname) if filename[-4:] == '.dat']
    if nbmax is not None:
        etdexlist = etdexlist[:nbmax]
    return (Etdex(os.path.join(dirname,filename)) for filename in etdexlist)


# ----------------------------------------------------------------------------
def dataiterator(dirname,nbmax=None):
    """
    ETDEX.ITERDATA - Un itérateur qui renvoie des dataframes.

    :param dirname: Le répertoire contenant les fichiers ETDEX
    :param nbmax: un critère d'arrêt optionnel.
    :return: l'itérateur renvoie des DataFrames un par un.
    """
    etdexlist = [filename[:-4] + '.xml' for
                 filename in os.listdir(dirname) if filename[-4:] == '.dat']
    if nbmax is not None:
        etdexlist = etdexlist[:nbmax]
    return (Etdex(os.path.join(dirname, filename)).data() for filename in
            etdexlist)


# ----------------------------------------------------------------------------
def to_hdf5(dirname,storename,nbmax=None):
    """
    ETDEX.TO_HDF5 - Crée un store HDF5 à partir d'un répertoire de fifiers ETDEX.
    
    Ne gère que les transitoires.
    Les enregistrements sont mis à la suite dans le HDF5Store sous la forme 
    de DataFrame pandas.
    
    Renvoie le nombre d'enregistrements écrits.
    """
    
    # Destruction de l'ancien store s'il existe.
    with pd.HDFStore(storename,mode='w') as store:
        n = 0
        for df in dataiterator(dirname,nbmax):
            store[df.index.name] = df
            n += 1
    return n
         

#%% Exécution pour le debuggage
if __name__ == "__main__":
    filename1 = "BCIAM_C1A__TR__DEV__d20070907_h124025__17.xml"
    filename2 = "BPOL_OATB__TR__DEV__d20070601_h121558__100.xml"
    filename3 = "BPOL_OATB__TR__DEV__d20070601_h124526__5.xml"
    filename4 = "BPOL_OATB__TR__DEV__d20070618_h212538__15.xml"
    filename5 = "BPOL_OATB__TR__DEV__d20071205_h135230__.xml"
    
    dirxml = os.environ['HOME']+"wrk/bancs/data/sam146/"
    direxp = os.environ['HOME']+"wrk/bancs/extract/"
    sname = "extrait.h5"

    test_read = False
    test_data = False
    test_freq = False
    test_iter = False
    test_hdf5 = True
    
    if test_read or test_data:    # Chargement des données.
        print("\n*Lecture de fichiers originaux*")
        e1 = Etdex(dirxml+filename1)  # fréquence unique
        print(e1)
        e2 = Etdex(dirxml+filename2)  # fréquence non unique
        print(e2)
        e3 = Etdex(dirxml+filename3)  # 2 points
        print(e3)
        e4 = Etdex(dirxml+filename4)  # 1 point
        print(e4)
        e5 = Etdex(dirxml+filename5)  # 1 point
        print(e5)
        
        if test_data:   # Test de lecture des données
            print("\n*Test de lecture des données*")
            df1 = e1.data() ; print("E1 :",df1.shape)
            df2 = e2.data() ; print("E2 :",df2.shape)
            df3 = e3.data() ; print("E3 :",df3.shape)
            df4 = e4.data() ; print("E4 :",df4.shape)
            df5 = e5.data() ; print("E5 :",df5.shape)

    if test_freq:   # Cas des petites feréquences.
        print('\n*lecture des petites fréquences*')
        filename3 = "BPOL_OATB__TR__DEV__d20070601_h124526__5.xml"
        dirname = os.environ['HOME']+"wrk/bancs/data/sam146/"
        e3 = Etdex(dirname+filename3)
        print(e3)

    if test_iter:   # Test de l'itérateur.
        print("\n*Test de l'itération*")
        print("Nombre total de transitoires :",nbtransient(dirxml))
        for e in iterator(dirxml,nbmax=3):
            print(e.filename)
    
    if test_hdf5: # Test du stockge de base des données.
        to_hdf5(dirxml,os.path.join(direxp,sname),nbmax=5)
