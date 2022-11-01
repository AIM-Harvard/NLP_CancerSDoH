## Data Preprocessing for Log Odds Ratio Analysis

The first step to data preprocessing is to add dataset paths to the Person_Names.py file (lines 51 and 57). The file creates three lists containing all patient names, provider names, and other popular person names. If there are no patient or provider names provided in the dataset, you can run only lines 1-35 of this file to create a list of popular person names (other_names), as identified by the names_dataset package. These lists of names will be used in Data_Preprocessing.py to identify which words to mask with 'NAME'.


The code used to process raw text columns is Data_Preprocessing.py. This code contains the Clean_Note_Columns function which adds the following three columns to a dataset:

Col 1: Stopwords, punctuation, digits removed with NAME masking

Col 2:  Stopwords, punctuation, digits removed with NAME/GENDER/PRONOUN/HONORIFIC masking

Col 3: Stopwords, punctuation, digits removed with LANGUAGE/RACE/NATIONALITY/RELIGION/NAME masking


The Medical_Nonmedical_Note_Split.py file can be used to extract the medical and nonmedical words from cleaned text, according to their UMLS codes. This file contains the medical_nonmedical function which takes as input a pandas dataframe containing cleaned text and outputs the same dataframe with two new columns: one containing only the medical words in the text column and one containing only the nonmedical words in the text column
