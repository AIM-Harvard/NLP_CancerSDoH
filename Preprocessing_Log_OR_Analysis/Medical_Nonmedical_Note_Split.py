# -*- coding: utf-8 -*-

"""
This file contains the medical_nonmedical function which takes as input a dataframe containing a column of 
text and returns a dataframe with two new columns: one containing only the medical words in the text column and 
one containing only the nonmedical words in the text column. Medical and non-medical words were identified 
according to their Unified Medical Language System (UMLS) semantic type. 

There are 133 semantic types in the UMLS repository, and 50 of these were categorized as non-medical by a 
data scientist and oncologist with expertise in NLP. If a word had both medical and nonmedical semantic types,
the word was categorized as non-medical.

Any word that did not have a UMLS semantic type was excluded from both the medical and non-medical columns. 
"""

# Load required packages
import numpy as np
import spacy  
spacy.prefer_gpu()
import pandas as pd

import scispacy
import en_core_sci_sm
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking import EntityLinker
from spacy.language import Language
from scispacy.candidate_generation import CandidateGenerator
from scispacy.umls_utils import UmlsKnowledgeBase
kb = UmlsKnowledgeBase()
st = kb.semantic_type_tree
    
nlp = spacy.load("en_core_sci_sm")
config = {"resolve_abbreviations": True,"linker_name": "umls"}    # Maybe get rid of the max_entities_per_mention?
nlp.add_pipe("scispacy_linker", config=config)
candidate_generator = CandidateGenerator(name="umls")
entity_linker = nlp.get_pipe('scispacy_linker')
entity_linker.candidate_generator = candidate_generator
entity_linker.kb = candidate_generator.kb



"""
Define all lists that were used for masking to ensure these don't end up in the medical column
We will combine all of these lists into a set called `nonmedical_set`.

Racial categories were chosen from the census (added 'Native American' and 'Indigenous' as well as 'Caucasian')

Religions were obtained from wikipedia: https://en.wikipedia.org/wiki/List_of_religions_and_spiritual_traditions

A list of (hopefully all) nationalities was obtained from github: https://gist.github.com/zamai/8e0d30a7f23f33e9c220c20db71c80dc

Some of the world's most popular languages were obtained here: https://www.berlitz.com/blog/most-spoken-languages-world

Pronouns obtained here: https://www.thefreedictionary.com/List-of-pronouns.htm#:~:text=Pronouns%20are%20classified%20as%20personal,%2C%20yours%2C%20his%2C%20hers%2C

List of common (but not all) genders obtained here: https://www.medicalnewstoday.com/articles/types-of-gender-identity#types-of-gender-identity

List of common honorifics obtained from here: https://en.wikipedia.org/wiki/English_honorifics#Common_titles

"""

Racial_Categories = ["american indian", "native american", "alaskan native", "indigenous", "asian", "black", "african american", "hispanic", "latino", "native hawaiin", "pacific islander","white","caucasian", "aa", "af amr"]
Religion_Categories = ["christian","muslim","atheist","agnostic","hindu","buddhist", "sikh", "jewish", "jain", "shintoist", "pagan", "rastafarian"] 
Nationalities_UC = ['Afghan', 'Albanian', 'Algerian', 'American', 'Andorran', 'Angolan', 'Antiguans', 'Argentinean', 'Armenian', 'Australian', 'Austrian', 'Azerbaijani', 'Bahamian', 'Bahraini', 'Bangladeshi', 'Barbadian', 'Barbudans', 'Batswana', 'Belarusian', 'Belgian', 'Belizean', 'Beninese', 'Bhutanese', 'Bolivian', 'Bosnian', 'Brazilian', 'British', 'Bruneian', 'Bulgarian', 'Burkinabe', 'Burmese', 'Burundian', 'Cambodian', 'Cameroonian', 'Canadian', 'Cape Verdean', 'Central African', 'Chadian', 'Chilean', 'Chinese', 'Colombian', 'Comoran',  'Congolese', 'Costa Rican', 'Croatian', 'Cuban', 'Cypriot', 'Czech', 'Danish', 'Djibouti', 'Dominican', 'Dutch', 'Dutchman', 'Dutchwoman', 'East Timorese', 'Ecuadorean', 'Egyptian', 'Emirian', 'Equatorial Guinean', 'Eritrean', 'Estonian', 'Ethiopian', 'Fijian', 'Filipino', 'Finnish', 'French', 'Gabonese', 'Gambian', 'Georgian', 'German', 'Ghanaian', 'Greek', 'Grenadian', 'Guatemalan', 'Guinea-Bissauan', 'Guinean', 'Guyanese', 'Haitian', 'Herzegovinian', 'Honduran', 'Hungarian', 'I-Kiribati', 'Icelander', 'Indian', 'Indonesian', 'Iranian', 'Iraqi', 'Irish', 'Israeli', 'Italian', 'Ivorian', 'Jamaican', 'Japanese', 'Jordanian', 'Kazakhstani', 'Kenyan', 'Kittian and Nevisian', 'Kuwaiti', 'Kyrgyz', 'Laotian', 'Latvian', 'Lebanese', 'Liberian', 'Libyan', 'Liechtensteiner', 'Lithuanian', 'Luxembourger', 'Macedonian', 'Malagasy', 'Malawian', 'Malaysian', 'Maldivan', 'Malian', 'Maltese', 'Marshallese', 'Mauritanian', 'Mauritian', 'Mexican', 'Micronesian', 'Moldovan', 'Monacan', 'Mongolian', 'Moroccan', 'Mosotho', 'Motswana', 'Mozambican', 'Namibian', 'Nauruan', 'Nepalese', 'Netherlander', 'New Zealander', 'Ni-Vanuatu', 'Nicaraguan', 'Nigerian', 'Nigerien', 'North Korean', 'Northern Irish', 'Norwegian', 'Omani', 'Pakistani', 'Palauan', 'Panamanian', 'Papua New Guinean', 'Paraguayan', 'Peruvian', 'Polish', 'Portuguese', 'Qatari', 'Romanian', 'Russian', 'Rwandan', 'Saint Lucian', 'Salvadoran', 'Samoan', 'San Marinese', 'Sao Tomean', 'Saudi', 'Scottish', 'Senegalese', 'Serbian', 'Seychellois', 'Sierra Leonean', 'Singaporean', 'Slovakian', 'Slovenian', 'Solomon Islander', 'Somali', 'South African', 'South Korean', 'Spanish', 'Sri Lankan', 'Sudanese', 'Surinamer', 'Swazi', 'Swedish', 'Swiss', 'Syrian', 'Taiwanese', 'Tajik', 'Tanzanian', 'Thai', 'Togolese', 'Tongan', 'Tribe', 'Tribal', 'Trinidadian or Tobagonian', 'Tunisian', 'Turkish', 'Tuvaluan', 'Ugandan', 'Ukrainian', 'Uruguayan', 'Uzbekistani', 'Venezuelan', 'Vietnamese', 'Welsh', 'Yemenite', 'Zambian', 'Zimbabwean']  
Nationalities = [nat.lower() for nat in Nationalities_UC]   
Languages = ["english","spanish", "portuguese", "mandarin", "hindi", "french", "arabic", "bengali", "russian", "indonesian"] 
Pronouns = ["he", "she", "they", "her", "him", "them","hers", "his", "theirs", "her", "his", "their", "herself", "himself", "themselves"] 
Genders = ["male", "female", "man", "gentleman", "woman", "lady", "girl", "boy", "agender", "androgyne", "bigender", "butch", "cisgender", "genderfluid", "gender outlaw", "genderqueer", "masculine of center", "nonbinary", "omnigender", "polygender", "transgender","trans", "two spirit"] 
Honorifics = ["mr", "mrs", "miss", "ms", "sir", "mister"] 


nonmedical_set = set(Racial_Categories + Religion_Categories + Nationalities + Languages + Pronouns + Genders + Honorifics)

# African American was not being placed in the nonmedical column (since 'african' did not have a umls semantic code and was not in the nonmedical_set)
# To fix this, add 'african' to the nonmedical set and also add 'af' and 'amr' to the nonmedical set
nonmedical_set.add("african")
nonmedical_set.add("af")
nonmedical_set.add("amr")

# Define nonmedical stycodes
nonmed_sentiment_codes = {"T073", "T078", "T070","T169", "T080", "T079", "T056", "T054", "T041", "T170", "T167", "T098", "T002", "T097", "T083", "T092", "T204", "T168", "T016", "T069", "T055", "T099", "T013", "T067", "T008", "T068", "T015", "T057", "T090", "T014", "T101", "T096", "T065", "T064", "T071", "T012","T072", "T077", "T093", "T051", "T100","T052","T171","T089","T011","T053","T095","T066","T102",""}



"""
Create function that takes as input a dataframe and string column to edit. utputs the same dataframe with two new columns containing all the medical and all the nonmedical words in each note.
    """


def medical_nonmedical(df, col_to_edit, nonmedical_stycodes):
    
    """
    df: pandas dataframe
    col_to_edit: column of notes from which to extract medical and nonmedical words
    nonmedical_stycodes: list of semantic types to consider nonmedical
    
    returns: pandas dataframe containing all the columns in df along with a col_to_edit_medical column, containing all the medical words in each note, and a col_to_edit_nonmedical column which contains all the nonmedical
    words in each note.
    """
    
        
    # The packages required for this function are right above this function. Trying to figure out if loading these packages each time is what's taking so long
    medical_col = []
    nonmedical_col = []
    
    for note in df[col_to_edit]:
        kept_words_medical = []
        kept_words_nonmedical = []
        if str(type(note)) == "<class 'str'>":  # Some of the rows were missing notes and instead contained NaN. Only execute the following code for the notes of type 'string'
            
 
            doc = nlp(note)
            
            for token in doc:
                if token.text.find(" ") == -1 and len(token.text) > 1:
                    if token.text[0] == "-" or token.text[0] == "." or token.text[0] == "/":
                        text = token.text[1:len(token.text)]
                    elif token.text[len(token.text)-1] == "-" or token.text[len(token.text)-1] == "." or token.text[len(token.text)-1] == "/":
                        text = token.text[0:len(token.text)-1]
                    else:
                        text = token.lemma_.lower()
                        
                        
                    if token.text in {"NAME", "GENDER", "PRONOUN", "HONORIFIC", "RACE", "LANGUAGE", "NATIONALITY", "RELIGION"}:
                        kept_words_nonmedical.append(token.text)
                    
                        
                    elif not text.lower() in nonmedical_set:
                        try:
      
                            cuis = kb.alias_to_cuis[text]    
                            cuis = entity_linker.kb.alias_to_cuis[text]
                            append_to_nonmedical = 0
                            for cui in cuis:
                                umls_ent = kb.cui_to_entity[cui]
                                
                                stycodes = [(stycode, st.get_canonical_name(stycode)) 
                                                for stycode in umls_ent.types]
                            
                                
                                for stycode, styname in stycodes:
                                    if stycode in nonmedical_stycodes:
                                        append_to_nonmedical = 1
                            if append_to_nonmedical == 1:
                                kept_words_nonmedical.append(text.lower())
                            else:
                                kept_words_medical.append(text.lower())
                                
        
                        except:
                            continue
                            
                    else:
                        kept_words_nonmedical.append(text.lower())
                        
            note_medical = " ".join([word for word in kept_words_medical])
            note_nonmedical = " ".join([word for word in kept_words_nonmedical])
            
            medical_col.append(note_medical)
            nonmedical_col.append(note_nonmedical)
            
        else:
            medical_col.append(np.nan)
            nonmedical_col.append(np.nan)
                
    
    df[col_to_edit + "_medical"] = medical_col
    df[col_to_edit + "_nonmedical"] = nonmedical_col

    return(df)

