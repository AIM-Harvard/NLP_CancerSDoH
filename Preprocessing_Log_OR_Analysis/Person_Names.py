# -*- coding: utf-8 -*-
"""
This files creates three sets of names: patient names, provider names, and other names. If using a dataset
that does not include columns for patient and provider names, only the other_names set can be used.

other_names include common english, spanish, chinese, italian, french, and nigerian names as identified
by the NameDataset package. 
"""

# Load required packages
import pandas as pd
from names_dataset import NameDataset
nd = NameDataset()


# Load common spanish, english, chinese, italian, french, and nigerian names from the NameDataset package
top_spanish_names = nd.get_top_names(n=500, country_alpha2 = "ES")
top_spanish_names = set(top_spanish_names.get("ES").get("M") + top_spanish_names.get("ES").get("F"))

top_english_names = nd.get_top_names(n=1000, country_alpha2 = "US")
top_english_names = set(top_english_names.get("US").get("M") + top_english_names.get("US").get("F"))

top_chinese_names = nd.get_top_names(n=500, country_alpha2 = "CN")
top_chinese_names = set(top_chinese_names.get("CN").get("M") + top_chinese_names.get("CN").get("F"))

top_italian_names = nd.get_top_names(n=500, country_alpha2 = "IT")
top_italian_names = set(top_italian_names.get("IT").get("M") + top_italian_names.get("IT").get("F"))

top_french_names = nd.get_top_names(n=500, country_alpha2 = "FR")
top_french_names = set(top_french_names.get("FR").get("M") + top_french_names.get("FR").get("F"))

top_nigerian_names = nd.get_top_names(n=500, country_alpha2 = "NG")
top_nigerian_names = set(top_nigerian_names.get("NG").get("M") + top_nigerian_names.get("NG").get("F"))

other_names = set.union(top_spanish_names, top_english_names, top_italian_names, top_french_names, top_nigerian_names, top_chinese_names)

# Remove words from other_names that rarely referred to person names in this dataset (optional)
# other_names.remove("Pretty")
# other_names.remove("Hair")
# other_names.remove("Cc")
# other_names.remove("April")
# other_names.remove("May")
# other_names.remove("June")
# other_names.remove("Jan")
# other_names.remove("Led")
# other_names.remove("Max")
# other_names.remove("He")


# Load the dataset containing patient names
merged_db = pd.read_csv("<Path_to_data>")
patient_first_names = merged_db["PatientFirstNM"]
patient_last_names = merged_db["PatientLastNM"]
patient_names = set(list(patient_first_names) + list(patient_last_names))

# Load the dataset containing provider names
provider_name_df = pd.read_csv("<Path_to_data>")

# In this dataset, provider names were listed as "<LASTNAME>, <FIRSTNAME> <DEGREES>". For example,
# an entry may look like "Smith, John M.D. PH.D.". Create two empty lists in which to store the 
# extracted first and last names. 

provider_first_names = []
provider_last_names = []
for i in range(len(provider_name_df)):
    current_line = provider_name_df[0][i]
    provider_last_names.append(current_line[0: current_line.find(",")].lower())
    next_line = current_line[current_line.find(",") + 2: len(current_line)]
    if next_line.find(" ") > -1:

        provider_first_names.append(next_line[0:next_line.find(" ")].lower())

    else:
        provider_first_names.append(next_line[0:len(next_line)].lower())
        
provider_names = set(provider_first_names + provider_last_names)

# If tokenization will split words containing a dash (-), run the code below to add the first and last
# half of hyphenated names as two separate names to the provider_names set
# names_to_drop = set()
# for name in provider_names:
#     if name.find("-") > -1:
#         provider_names = set(list(provider_names) + name.split("-"))
#     elif name.find(" ") > -1:
#         provider_names = set(list(provider_names) + name.split(" "))
#     elif name.find(".") > -1 or len(name) == 1 or name.find(",") > -1:
#         names_to_drop.add(name)
#         #provider_names.remove(name)
#     elif name == "":
#         names_to_drop.add(name)
#         #provider_names.remove(name)
#     else:
#         continue

# for name in names_to_drop:
#     provider_names.remove(name)

# Remove strange provider names that rarely referred to person names in the dataset (optional)
# provider_names.remove("child")
# provider_names.remove("moment")
# provider_names.remove("see")
# provider_names.remove("do")
# provider_names.remove("st")
# provider_names.remove("main")
# provider_names.remove("brain")
# provider_names.remove("large")
# provider_names.remove("low")
# provider_names.remove("hurt")
# provider_names.remove("day")
# provider_names.remove("come")
# provider_names.remove("min")
# provider_names.remove("sober")
# provider_names.remove("friend")
# provider_names.remove("march")
# provider_names.remove("small")
# provider_names.remove("body")
# provider_names.remove("deep")
# provider_names.remove("glass")
# provider_names.remove("md")
# provider_names.remove("certain")
# provider_names.remove("orifice")
# provider_names.remove("ring")
# provider_names.remove("close")
# provider_names.remove("gone")
# provider_names.remove("pace")
# provider_names.remove("mount")
# provider_names.remove("np")
# provider_names.remove("shorten")
# provider_names.remove("finger")
# provider_names.remove("long")
# provider_names.remove("back")
# provider_names.remove("hazard")
# provider_names.remove("bueno")
# provider_names.remove("battle")
# provider_names.remove("beers")
# provider_names.remove("block")
# provider_names.remove("tone")
# provider_names.remove("case")
# provider_names.remove("rn")
# provider_names.remove("hope")
# provider_names.remove("box")
# provider_names.remove("parent")
# provider_names.remove("bean")
# provider_names.remove("bath")
# provider_names.remove("butter")
# provider_names.remove("generic")
# provider_names.remove("mask")
# provider_names.remove("fix")
# provider_names.remove("hunt")
# provider_names.remove("forget")
# provider_names.remove("constant")
# provider_names.remove("chew")
# provider_names.remove("sink")
# provider_names.remove("palm")
# provider_names.remove("mail")
# provider_names.remove("brick")
# provider_names.remove("stump")
# provider_names.remove("settle")
# provider_names.remove("damp")
# provider_names.remove("strange")
# provider_names.remove("ten")
# provider_names.remove("few")
# provider_names.remove("strong")
# provider_names.remove("cone")
# provider_names.remove("friday")
# provider_names.remove("seed")
# provider_names.remove("sweet")
# provider_names.remove("pa")
# provider_names.remove("lock")
# provider_names.remove("straw")
# provider_names.remove("kit")
# provider_names.remove("bold")
# provider_names.remove("dec")
# provider_names.remove("fee")
# provider_names.remove("nov")
# provider_names.remove("fish")
# provider_names.remove("ng")
# provider_names.remove("driver")


    
    
