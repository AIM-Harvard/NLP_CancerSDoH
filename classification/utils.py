
"""
Helper Functions for the SDOH Bias Project
"""

import pandas as pd
import regex as re
import ast
import glob
import numpy as np
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep

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
