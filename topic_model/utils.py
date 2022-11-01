
"""
Helper Functions for the SDOH Bias Project
"""

import pandas as pd
import regex as re
import ast
import glob
import numpy as np


GENDERED_DISEASE_SITES = ['GU-Prostate',
 'Breast-NOS',
 'Gyn-Uterus',
 'Breast-DCIS',
 'Breast',
 'Gyn-Cervix',
 'Gyn-Vulva',
 'Gyn-NOS',
 'Gyn-Ovary',
 'GU-Testicular',
 'Gyn-Vagina',
 'GU-Penis',
 'Breast-Gynecomastia']

def load_data(path:str) -> pd.DataFrame:
    """
    path: a path to the directory containing csv files
    returns: pandas dataframe of patient notes
    NOTE: disease site in broadband are lists and must be processed with ast.literal_eval, example usage in filter_data
    columns are [PMRN, DFCI_MRN, Race, Race_group, Gender, disease_site, RPT_STATUS, text]
    """
    all_files = glob.glob(path + "/*.csv")
    df_lst = list()
    for corp in all_files:
        df = pd.read_csv(corp, encoding='utf8')
        df.reset_index(inplace=True, drop=True)
        df_lst.append(df)
    
    corpus = pd.concat(df_lst)
    corpus = corpus[corpus['RPT_STATUS'].isin(['Signed', 'Addendum'])]
    corpus.reset_index(inplace=True, drop=True)
    return corpus

def filter_data(data:pd.DataFrame, col:str, exclusions:list, include:bool = False) -> list:
    """
    Params
    -data: the pandas dataframe of the corpus
    -col: the column header to apply the filter
    -exlcusions: a list values within the column to exlcude from the dataframe
    -include: If "True", the list of "exclusions" will be treated in an opposite manner. I.e., the rows with values from the exlcusions list will be the ONLY items included and returned.
    
    NOTE: disease_site exclusions will remove PATIENTS that have disease site exlusions
    
    EXAMPLE: removing gendered disease sites
    data = pd.read_csv('/path/to/corpus/file.csv', encoding='utf8')
    col = disease_site
    exclusions = ["GU-Penis", "Gyn-Ovary"]
    """
    if col == 'disease_site':
        gendered_disease_patients = set()
        exlcus = set(exclusions)
        disease_sites = data[col]
        for i in range(len(data)):
            try:
                dis_sites = ast.literal_eval(disease_sites[i])
                for d_s in dis_sites:
                    if d_s in exlcus:
                        gendered_disease_patients.add(data.loc[i, 'PMRN'])
            except SyntaxError:
                pass
                # print(disease_sites[i])
        if include:
            filtered_data = data[data['PMRN'].isin(list(gendered_disease_patients))]
        else:
            filtered_data = data[~(data['PMRN'].isin(list(gendered_disease_patients)))]
    else:
        if include:
            filtered_data = data[data[col].isin(exclusions)]
        else:
            filtered_data = data[~(data[col].isin(exclusions))]
    filtered_data.reset_index(inplace=True, drop=True)

    return filtered_data

def grab_sections(text:str, token_len:int = 300) -> str:
    """
    text: emr note to find sections
    returns: some or all of Interval History, History of Present Illness, Assesment and Plan sections
    if none are found in the emr, the fucntion will return an empty string
    """
    if isinstance(text, str):
        ih_pat = '(interval\s+history|i\/h|ih:)'
        hpi_pat = '(history\s+of\s+present\s+illness|hpi|h.p.i)'
        ap_pat = '(assesment\s+and\s+plan|a/p|ap:)'

        interval_hist = re.search(ih_pat, text, re.IGNORECASE)
        hpi = re.search(hpi_pat, text, re.IGNORECASE)
        ap = re.search(ap_pat, text, re.IGNORECASE)

        hist_toks = []
        hpi_toks = []
        ap_toks = []
        if interval_hist:
            temp_hist = text[interval_hist.start():]
            temp_hist_toks = temp_hist.split()
            hist_toks = temp_hist_toks[:token_len]
        if hpi:
            temp_hpi = text[hpi.start():]
            temp_hpi_toks = temp_hpi.split()
            hpi_toks = temp_hpi_toks[:token_len]
        if ap:
            temp_ap = text[ap.start():]
            temp_ap_toks = temp_ap.split()
            ap_toks = temp_ap_toks[:token_len]

        out_str = ' '.join(hist_toks) +' '+ ' '.join(hpi_toks)+ ' '+ ' '.join(ap_toks)
        if len(out_str.split()) > 0:
            return out_str
        else:
            return ''
    return ''