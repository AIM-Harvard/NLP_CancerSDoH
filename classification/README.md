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
