Neural Topic Modeling of Clincal notes
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
