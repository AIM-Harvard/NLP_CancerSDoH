# -*- coding: utf-8 -*-

"""
This file contains the Clean_Note_Columns function which takes in a dataframe containing a column of raw 
(unprocessed) text and outputs the same dataframe with three new columns of cleaned text:
    
    
Col 1: Stopwords, punctuation, digits removed with NAME masking

Col 2:  Stopwords, punctuation, digits removed with NAME/GENDER/PRONOUN/HONORIFIC masking

Col 3: Stopwords, punctuation, digits removed with LANGUAGE/RACE/NATIONALITY/RELIGION/NAME masking


"""

### Load Required Packages ###
import numpy as np
import spacy  #This is needed to break sentences into tokens
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm") #Download a trained pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Person_Names import other_names   
from Person_Names import patient_names
from Person_Names import provider_names




'''
Create lists of words to use for masking

Racial categories were chosen from the census (added 'Native American' and 'Indigenous' as well as 'Caucasian')

Religions were obtained from wikipedia: https://en.wikipedia.org/wiki/List_of_religions_and_spiritual_traditions

A list of (hopefully all) nationalities was obtained from github: https://gist.github.com/zamai/8e0d30a7f23f33e9c220c20db71c80dc

Some of the world's most popular languages were obtained here: https://www.berlitz.com/blog/most-spoken-languages-world

Pronouns obtained here: https://www.thefreedictionary.com/List-of-pronouns.htm#:~:text=Pronouns%20are%20classified%20as%20personal,%2C%20yours%2C%20his%2C%20hers%2C

List of common (but not all) genders obtained here: https://www.medicalnewstoday.com/articles/types-of-gender-identity#types-of-gender-identity

List of common honorifics obtained from here: https://en.wikipedia.org/wiki/English_honorifics#Common_titles

'''

Racial_Categories = set(["american indian", "native american", "alaskan native", "indigenous", "asian", "black", "african american", "hispanic", "latino", "native hawaiin", "pacific islander","white","caucasian", "aa", "af amr"])
Religion_Categories = set(["christian","muslim","atheist","agnostic","hindu","buddhist", "sikh", "jewish", "jain", "shintoist", "pagan", "rastafarian"])
Nationalities_UC = ['Afghan', 'Albanian', 'Algerian', 'American', 'Andorran', 'Angolan', 'Antiguans', 'Argentinean', 'Armenian', 'Australian', 'Austrian', 'Azerbaijani', 'Bahamian', 'Bahraini', 'Bangladeshi', 'Barbadian', 'Barbudans', 'Batswana', 'Belarusian', 'Belgian', 'Belizean', 'Beninese', 'Bhutanese', 'Bolivian', 'Bosnian', 'Brazilian', 'British', 'Bruneian', 'Bulgarian', 'Burkinabe', 'Burmese', 'Burundian', 'Cambodian', 'Cameroonian', 'Canadian', 'Cape Verdean', 'Central African', 'Chadian', 'Chilean', 'Chinese', 'Colombian', 'Comoran',  'Congolese', 'Costa Rican', 'Croatian', 'Cuban', 'Cypriot', 'Czech', 'Danish', 'Djibouti', 'Dominican', 'Dutch', 'Dutchman', 'Dutchwoman', 'East Timorese', 'Ecuadorean', 'Egyptian', 'Emirian', 'Equatorial Guinean', 'Eritrean', 'Estonian', 'Ethiopian', 'Fijian', 'Filipino', 'Finnish', 'French', 'Gabonese', 'Gambian', 'Georgian', 'German', 'Ghanaian', 'Greek', 'Grenadian', 'Guatemalan', 'Guinea-Bissauan', 'Guinean', 'Guyanese', 'Haitian', 'Herzegovinian', 'Honduran', 'Hungarian', 'I-Kiribati', 'Icelander', 'Indian', 'Indonesian', 'Iranian', 'Iraqi', 'Irish', 'Israeli', 'Italian', 'Ivorian', 'Jamaican', 'Japanese', 'Jordanian', 'Kazakhstani', 'Kenyan', 'Kittian and Nevisian', 'Kuwaiti', 'Kyrgyz', 'Laotian', 'Latvian', 'Lebanese', 'Liberian', 'Libyan', 'Liechtensteiner', 'Lithuanian', 'Luxembourger', 'Macedonian', 'Malagasy', 'Malawian', 'Malaysian', 'Maldivan', 'Malian', 'Maltese', 'Marshallese', 'Mauritanian', 'Mauritian', 'Mexican', 'Micronesian', 'Moldovan', 'Monacan', 'Mongolian', 'Moroccan', 'Mosotho', 'Motswana', 'Mozambican', 'Namibian', 'Nauruan', 'Nepalese', 'Netherlander', 'New Zealander', 'Ni-Vanuatu', 'Nicaraguan', 'Nigerian', 'Nigerien', 'North Korean', 'Northern Irish', 'Norwegian', 'Omani', 'Pakistani', 'Palauan', 'Panamanian', 'Papua New Guinean', 'Paraguayan', 'Peruvian', 'Polish', 'Portuguese', 'Qatari', 'Romanian', 'Russian', 'Rwandan', 'Saint Lucian', 'Salvadoran', 'Samoan', 'San Marinese', 'Sao Tomean', 'Saudi', 'Scottish', 'Senegalese', 'Serbian', 'Seychellois', 'Sierra Leonean', 'Singaporean', 'Slovakian', 'Slovenian', 'Solomon Islander', 'Somali', 'South African', 'South Korean', 'Spanish', 'Sri Lankan', 'Sudanese', 'Surinamer', 'Swazi', 'Swedish', 'Swiss', 'Syrian', 'Taiwanese', 'Tajik', 'Tanzanian', 'Thai', 'Togolese', 'Tongan', 'Tribe', 'Tribal', 'Trinidadian or Tobagonian', 'Tunisian', 'Turkish', 'Tuvaluan', 'Ugandan', 'Ukrainian', 'Uruguayan', 'Uzbekistani', 'Venezuelan', 'Vietnamese', 'Welsh', 'Yemenite', 'Zambian', 'Zimbabwean']  
Nationalities = [nat.lower() for nat in Nationalities_UC]
Languages = set(["english","spanish", "portuguese", "mandarin", "hindi", "french", "arabic", "bengali", "russian", "indonesian"])
Pronouns = {"he", "she", "they", "her", "him", "them","hers", "his", "theirs", "her", "his", "their", "herself", "himself", "themselves"}
Genders = {"male", "female", "man", "gentleman", "woman", "lady", "girl", "boy", "agender", "androgyne", "bigender", "butch", "cisgender", "genderfluid", "gender outlaw", "genderqueer", "masculine of center", "nonbinary", "omnigender", "polygender", "transgender","trans", "two spirit"}
Honorifics = {"Mr", "Mrs", "Miss", "Ms", "Sir", "Mister"}
nonmedical_set = set(list(Racial_Categories) + list(Religion_Categories) + list(Nationalities) + list(Languages) + list(Pronouns) + list(Genders) + list(Honorifics))


    

#remove pronouns, genders, and honorifics from stopwords 
stopwords = set(stopwords.words())
for word in Pronouns:
    if word in stopwords:
        stopwords.remove(word)
for word in Genders:
    if word in stopwords:
        stopwords.remove(word)
for word in Honorifics:
    if word in stopwords:
           stopwords.remove(word)


def Clean_Note_Columns(df, col_to_edit):   
    """
    df: pandas dataframe containing a column of notes
    col_to_edit: name of the note column to use for cleaning
    returns: df with all of its original columns plus 9 new columns:
        Col 1: Stopwords, punctuation, digits removed with NAME masking
        Col 2:  Stopwords, punctuation, digits removed with NAME/GENDER/PRONOUN/HONORIFIC masking
        Col 3: Stopwords, punctuation, digits removed with LANGUAGE/RACE/NATIONALITY/RELIGION/NAME masking

    """

    # Create three empty lists in which to store the new notes
    result_col1 = []    
    result_col2 = []
    result_col3 = []
    
    for i in range(len(df[col_to_edit])):
        note = df[col_to_edit][i]
        if str(type(note)) == "<class 'str'>":
            
            # Create lists to store the kept words from each note
            kept_words_col1 = []
            kept_words_col2 = []
            kept_words_col3 = []
            
            # Start by removing all stopwords
            text_tokens = word_tokenize(note.lower())
            tokens_without_sw = [word for word in text_tokens if not word.lower() in stopwords and not word.isdigit() and not any(char.isdigit() for char in word)]
            new_note = ""
            for word in tokens_without_sw:
                new_note = new_note + word + " "
            note = new_note

            # Remove punctuation
            doc = nlp(note) 
            note = " ".join([t.text if not t.is_punct else "" for t in doc])
            doc = nlp(note) 
            
            
            for i in range(len(doc)):
                t = doc[i]
                
                if t.text.lower() in Pronouns:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col3.append(t.lemma_.lower())
                    kept_words_col2.append("PRONOUN")

                    
                elif t.text.lower() in Genders:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col3.append(t.lemma_.lower())
                    kept_words_col2.append("GENDER") 
                    
                    
                elif t.text.lower() in Honorifics:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col3.append(t.lemma_.lower())
                    kept_words_col2.append("HONORIFIC")

                    
                elif t.text.lower() == "african":
                    if doc[i+1].text.lower() == "american":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
                        kept_words_col3.append("RACE")
  
                        
                elif t.text.lower() == "american":
                    if doc[i-1].text.lower() == "african":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
       
                        
                elif t.text.lower() == "af":
                    if doc[i+1].text.lower() == "amr":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
                        kept_words_col3.append("RACE")
                        
                        
                elif t.text.lower() == "amr":
                    if doc[i-1].text.lower() == "af":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
        
                        
                elif t.text.lower() == "american":
                    if doc[i+1].text.lower() == "indian":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
                        kept_words_col3.append("RACE")
   
                        
                elif t.text.lower() == "indian":
                    if doc[i-1].text.lower() == "american":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
        
                        
                elif t.text.lower() == "native":
                    if doc[i+1].text.lower() == "american":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
                        kept_words_col3.append("RACE")
            
                        
                elif t.text.lower() == "american":
                    if doc[i-1].text.lower() == "native":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
   
                        
                elif t.text.lower() == "alaskan":
                    if doc[i+1].text.lower() == "native":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
                        kept_words_col3.append("RACE")
     
                        
                elif t.text.lower() == "native":
                    if doc[i-1].text.lower() == "alaskan":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
 
                        
                elif t.text.lower() == "native":
                    if doc[i+1].text.lower() == "hawaiian":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
                        kept_words_col3.append("RACE")

                        
                elif t.text.lower() == "hawaiian":
                    if doc[i-1].text.lower() == "native":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
             
                        
                elif t.text.lower() == "pacific":
                    if doc[i+1].text.lower() == "islander":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
                        kept_words_col3.append("RACE")
            
                        
                elif t.text.lower() == "islander":
                    if doc[i-1].text.lower() == "pacific":
                        kept_words_col1.append(t.lemma_.lower())
                        kept_words_col2.append(t.lemma_.lower())
           
                        
                        
                elif t.text.lower() in Racial_Categories:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col2.append(t.lemma_.lower())
                    kept_words_col3.append("RACE")

                    
                elif t.text.lower() in Languages:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col2.append(t.lemma_.lower())
                    kept_words_col3.append("LANGUAGE")
                    
                elif t.text.lower() in Nationalities:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col2.append(t.lemma_.lower())
                    kept_words_col3.append("NATIONALITY")

                
                elif t.text.lower() in Religion_Categories:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col2.append(t.lemma_.lower())
                    kept_words_col3.append("RELIGION")
 
                    
                elif t.text.title() in other_names or t.text.title() in patient_names or t.text.lower() in provider_names:
                    if t.text.lower() not in nonmedical_set:
                        kept_words_col1.append("NAME")
                        kept_words_col2.append("NAME")
                        kept_words_col3.append("NAME")
  
                    
                
                else:
                    kept_words_col1.append(t.lemma_.lower())
                    kept_words_col2.append(t.lemma_.lower())
                    kept_words_col3.append(t.lemma_.lower())
        
                    
                    
            
            note1 = " ".join([word for word in kept_words_col1])
            note2 = " ".join([word for word in kept_words_col2])
            note3 = " ".join([word for word in kept_words_col3])
 
            
            result_col1.append(note1)
            result_col2.append(note2)
            result_col3.append(note3)
  
            
            
        else:
            result_col1.append(np.nan)
            result_col2.append(np.nan)
            result_col3.append(np.nan)
            
    
    
    df[col_to_edit + "_Cleaned1"] = result_col1
    df[col_to_edit + "_Cleaned2"] = result_col2
    df[col_to_edit + "_Cleaned3"] = result_col3
    
    return(df)


