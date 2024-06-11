# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:10:56 2024

@author: Ron Simenhois
"""

import os
from glob import glob
import pandas as pd

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), '../data'))
avalanche_data = os.path.join(DATA_PATH, 'CAIC_HWY_avalanches_2009-01-01_2021-05-04.csv')

def load_avalanches_by_patrial_path_name(path,
                                         path_name,
                                         **kwargs):
    """
    This fuction fetch all avalanches where the path name contains the string 
    in the "path_name" veriabole

    Parameters
    ----------
    path : str
        path to a csv file with the avalanche data.
    path_name : str
        path name or partial path name to setch the data file with.
    **kwargs : dict
        othe parameteres like avalanche size and type, trigger...
        to filter the search.

    Returns
    -------
    df : Pandas dataframe
        dataframe with the avalanche records after filtering.

    """
    
    av_types = kwargs.get('type', ['HS', 'SS'])
    av_trigs = kwargs.get('trig', ['N'])
    
    path_ids = list('ABCDE1234567')
    df = pd.read_csv(path, usecols=['Date', 'HW Path', 'Dsize','Trigger', 'Type', 'Elev', 'Terminus', 'Center Line Length'])
    df.loc[:, 'Trigger'] = df.loc[:, 'Trigger'].str.strip()
    df.loc[:, 'Type'] = df.loc[:, 'Type'].str.strip()
    df = df[(df['HW Path'].str.contains(path_name)) & (df['Trigger'].isin(av_trigs)) & (df['Type'].isin(av_types))] 
    df = df[df['HW Path'].str.contains('|'.join(path_ids))]
    if path_name == 'Sister':
        df = df.query('`HW Path` != "Seven Sister #6" or Elev == ">TL"')

    return df


def load_avalanches_by_paths_list(path,
                                  *path_names,
                                  **kwargs):
    """
    This fuction fetch all avalanches with avalanche path in the "path_names"
    tuple.

    Parameters
    ----------
    path : str
        path to a csv file with the avalanche data.
    path_name : str
        path name or partial path name to setch the data file with.
    **kwargs : dict
        othe parameteres like avalanche size and type, trigger...
        to filter the search.

    Returns
    -------
    df : Pandas dataframe
        dataframe with the avalanche records after filtering.

    """
    
    av_types = kwargs.get('type', ['HS', 'SS'])
    av_trigs = kwargs.get('trig', ['N'])
    
    df = pd.read_csv(path, usecols=['Date', 'HW Path', 'Dsize','Trigger', 'Type', 'Elev', 'Terminus', 'Center Line Length'])
    df.loc[:, 'Trigger'] = df.loc[:, 'Trigger'].str.strip()
    df.loc[:, 'Type'] = df.loc[:, 'Type'].str.strip()
    df = df[(df['HW Path'].isin(path_names)) & (df['Trigger'].isin(av_trigs)) & (df['Type'].isin(av_types))] 

    return df



if __name__ == '__main__':
    
    Sisters_vlancheces = load_avalanches_by_patrial_path_name(avalanche_data, 'Sister', **{'type': ['HS', 'SS'], 'trig': ['N']})
    print(Sisters_vlancheces[Sisters_vlancheces['Terminus'].isin(['BP', 'MP'])].groupby('HW Path').count())
    
    
    paths = ['Muleshoe', 'Telescope', 'Eagle', 'Porcupine'] 
    Eagle_groupe_avalanches = load_avalanches_by_paths_list(avalanche_data, *paths, **{'type': ['HS', 'SS'], 'trig': ['N']})
