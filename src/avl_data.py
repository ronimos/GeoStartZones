# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:10:56 2024

@author: Avalanche
"""

import os
from glob import glob
import pandas as pd

DATA_PATH = os.path.join(os.getcwd(), '../data/avalanches')
avalanche_data = os.path.join(DATA_PATH, 'CAIC_HWY_avalanches_2009-01-01_2021-05-04.csv')

def load_avalanches(path,
                    path_name,
                    **kwargs):
    
    av_types = kwargs.get('type', ['HS', 'SS'])
    av_trigs = kwargs.get('trig', ['N'])
    
    path_ids = list('ABCDE1234567')
    df = pd.read_csv(path, usecols=['Date', 'HW Path', 'Dsize','Trigger', 'Type', 'Elev', 'Terminus', 'Center Line Length'])
    df.loc[:, 'Trigger'] = df.loc[:, 'Trigger'].str.strip()
    df.loc[:, 'Type'] = df.loc[:, 'Type'].str.strip()
    df = df[(df['HW Path'].str.contains(path_name)) & (df['Trigger'].isin(av_trigs)) & (df['Type'].isin(['HS', 'SS']))] 
    df = df[df['HW Path'].str.contains('|'.join(path_ids))]
    if path_name == 'Sister':
        df = df.query('`HW Path` != "Seven Sister #6" or Elev == ">TL"')

    return df

if __name__ == '__main__':
    
    df = load_avalanches(avalanche_data, 'Sister', **{'type': ['HS', 'SS'], 'trig': ['N']})
    df[df['Terminus'].isin(['BP', 'MP'])].groupby('HW Path').count()
