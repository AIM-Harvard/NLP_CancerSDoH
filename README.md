# NLP Methods for Exporing Social Determinants of Health
***This repo is still in progress***

The goal of this project is to explore methods to empirically explore social determinants of health (SDoH) that may underlie cancer disparities. This code was used to implement the methods reported in "Natural language processing methods to empirically explore social determinants of health underlying cancer disparities" (manuscript in process). The results reported in this manuscript were from a prioprietary dataset with protected health information. But, these methods will run on other text datasets.

## Data Preprocessing for Log Odds Ratio Analysis

The first step to data preprocessing is to add dataset paths to the Person_Names.py file (lines 51 and 57). The file creates three lists containing all patient names, provider names, and other popular person names. If there are no patient or provider names provided in the dataset, you can run only lines 1-35 of this file to create a list of popular person names (other_names), as identified by the names_dataset package. These lists of names will be used in Data_Preprocessing.py to identify which words to mask with 'NAME'.


The code used to process raw text columns is Data_Preprocessing.py. This code contains the Clean_Note_Columns function which adds the following three columns to a dataset:

Col 1: Stopwords, punctuation, digits removed with NAME masking

Col 2:  Stopwords, punctuation, digits removed with NAME/GENDER/PRONOUN/HONORIFIC masking

Col 3: Stopwords, punctuation, digits removed with LANGUAGE/RACE/NATIONALITY/RELIGION/NAME masking


The Medical_Nonmedical_Note_Split.py file can be used to extract the medical and nonmedical words from cleaned text, according to their UMLS codes. This file contains the medical_nonmedical function which takes as input a pandas dataframe containing cleaned text and outputs the same dataframe with two new columns: one containing only the medical words in the text column and one containing only the nonmedical words in the text column

## Log Odds Ratios with Dirichlet Priors

The Create_Word_Lists.ipynb creates lists of all medical and nonmedical words that occur in patient notes, stratified by patient race. For example, if the word "intestinal" occurred 1408 times across all White patient notes, that word will also appear 1408 times in the list of medical words for White patients. These lists of words are used to compute the log odds ratios in Log_OR_Dirichlet_Prior_Race.ipynb


Log_OR_Dirichlet_Prior_Race.ipynb computes the log odds ratios to compare word frequency across patient race. It also computes the associated z-scores and confidence intervals. Non-hispanic White patient notes are compared to Black patient notes, and all other notes are compared to White patient notes.
