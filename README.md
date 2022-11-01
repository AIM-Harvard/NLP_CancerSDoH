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

## Neural Topic Modeling of Clincal notes
Topic Models are trained on corpus, and inference is performed on held out test set
Note Types: i) notes written only by physicians and; ii) notes written by all providers

Data and Models
    - Clincal notes are curated from MGB databases. All of our corpora contain PHI and are therefore not available to publish.
    - Variational Auto Encoder Topic Models
        - Purpose is to discover latent topics (word distributions) in our corpus of clincal notes
        - Then we perform inference using the trained topic models on clincial notes to see differences in topic distributions by patient demographic

Notebooks
- inference_neural.ipynb & inference_neural_phys_res_fel.ipynb 
    - All provider notes and only physician notes topic models respectively
    - Data is loaded and split into a training set and a held-out subset of notes for inference
    - Topic models are trained using automatic model selection from the Topic Modeling Neural Toolkit
    - Latent topics are displayed
    - Each cell can be run sequentially 

utils.py
    - Helper functions for classification

- Inference
    - Based on patient demographics (Insurance, Gender, and Race/Ethnicity) test set is encoded from trained topic model
    - Encodings are output to json for data analysis

## Clincal Note Patient Demographic Classifiers

Three binary classification tasks (Insurance, Gender, Race/Ethnicity) based on patient notes 
Note Types: i) notes written only by physicians and; ii) notes written by all providers

Data and Models
    - Clincal notes are curated from MGB databases. All of our corpora contain PHI and are therefore not available to publish
    - Training for BERT-base, Bio+Clincal-BERT, Logistic Regression, and Gradient Boosting models. 
    - After training and hyperparameter tuning evaluations are performed on the held out test set.

train.py
    - Python file to train and evaluate BERT-base and Bio+Clincal-BERT classifiers.
    - Load data, hyperparameters, and task arguments through command line arguments
    1. Data is loaded into memory and preprocessed 
    2. Model training, and evaluation on development set at each epoch
    3. Final test set evaluation
    EXAMPLE: Run BERT-base for 5 epochs, training on physician notes to classify a patient's gender
    e.g., python train.py --train_file /path/to/train_set.csv --dev_file /path/to/development_set.csv --test_file /path/to/test_set.csv --logdir /path/to/result_logs_dir --epochs 5  --model BERT --seq_length 512 --batch_size 32 --lr 0.00001 --label Gender --dropout 0.2 --undersample --provider_type Physician

utils.py
    - Helper functions for classification

Notebooks
- stat_model_all_providers.ipynb & stat_model_physician.ipynb
    - Trains and evaluates Logistic Regression and Gradient Boosting classifiers
    - Each cell can be run sequentially 
