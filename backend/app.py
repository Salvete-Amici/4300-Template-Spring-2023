import numpy as np
import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pickle5 as pickle
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = ""
MYSQL_PORT = 3306
MYSQL_DATABASE = "rep"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

with open('pickled_dict.pickle', 'rb') as handle:
    ml_output_dict = pickle.load(handle)


# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this

# global variables
mapping = None
ii = None
aii = None
URL = "https://www.food.com/"

allergies = {"nuts": ["peanut", "peanut oil", "hazelnut",
                      "peanut butter", "walnut", "pecan", "cashew",
                      "pistachio", "almond", "almond butter", "tree nut",
                      "macadamia", "pine nut", "pine", "brazil"],
             "vegetarian": ["bacon", "chicken", "beef", "duck", "fish",
                            "lamb, lobster", "oyster", "pork", "scallop",
                            "turkey", "venison", "crab", "clam", "rabbit",
                            "goat", "fowl", "sausage", "bologna", "chorizo",
                            "pepperoni", "tuna", "tilapia", "ham"],
             "dairy": ["milk", "butter", "ghee", "buttermilk",
                       "cheese", "paneer", "goat cheese", "sour cream",
                       "whipped cream", "half and half", "cottage cheese",
                       "whole milk", "ice cream", "pudding", "yogurt"],
             "gluten": ["flour", "pasta", "angel hair pasta", "rye", "wheat",
                        "bagel", "cereal", "beer"],
             "egg": ["egg", "eggs"],
             "shellfish": ["shrimp", "crab", "lobster", "prawn", "crawfish",
                           "crayfish", "crawdad", "clam", "cockle", "mussel",
                           "octopus", "oyster", "scallop", "sea cucumber", "snail"]
             }

allergies["vegan"] = allergies["vegetarian"] + \
    allergies["dairy"] + allergies["egg"]

time_lookup = {"Under 15 minutes" : [0,15], "Under 30 minutes": [0,30], "Under 1 hour": [0,60], "1-2 hours" : [60,120], "2+ hours" : [120,10000]}

category_lookup = {"Main Course" : ["main-dish", "soups-stews", "sandwiches", "breakfast"], "Appetizer and Snack" : ["appetizers", "side-dishes", "salads", "sauces", "condiments-etc"], "Dessert" : ["desserts"]}

def search_rank(query, optional, allergens, category, time, allergy_inverted_index, inverted_index, recipe_dict, time_lookup, category_lookup):
    # query is a list of strings, allergens is a list of strings, inverted_index is
    # a dictionary mapping ingredients to which recipes they appear in. recipe_dict
    # is a dictionary mapping recipe names to id, ingredients, and tags.
    # Returns a ranked list of the top 20 recipes
    if len(query) == 0:
        postings1 = list(recipe_dict.keys())
    else:
        postings1 = inverted_index[stemmer.stem(query[0].lower())]

    for ingredient in query:
        postings2 = inverted_index[stemmer.stem(ingredient.lower())]
        postings1 = merge_postings(postings1, postings2)

    for allergen in allergens:
        if allergen == "":
            break
        allergen_postings = allergy_inverted_index[allergen]
        postings1 = not_merge_postings(postings1, allergen_postings)
  
    category_postings = []
    for posting in postings1:
        if category == "":
            category_postings = postings1
            break
        if len(set(category_lookup[category]).intersection(set(recipe_dict[posting]['tags']))) > 0:
            category_postings.append(posting)

    time_postings = []
    for posting in category_postings:
        if time == "":
            time_postings = category_postings
            break
        time1 = time_lookup[time][0]
        time2 = time_lookup[time][1]
        if time1 <= recipe_dict[posting]['minutes'] and time2 >= recipe_dict[posting]['minutes']:
            time_postings.append(posting)

    similarity_ranking = []
    for posting in time_postings:
        ingredients = recipe_dict[posting]['ingredients']
        jaccard_sim = jaccard(
            list(set(query).union(set(optional))), ingredients)
        rating_score = get_rating(recipe_dict, posting)/5
        similarity_ranking.append((posting, jaccard_sim+rating_score))

    similarity_ranking.sort(reverse=True, key=lambda x: x[1])
    boundary = min(20, len(similarity_ranking))
    return similarity_ranking[:boundary]


def preprocess(recipe_list):
    # Returns a dictionary mapping recipe name to id, a list of ingredients, and a list of tags
    # Would be good to clean the recipe name
    recipe_dictionary = {}
    for recipe in recipe_list:
        name = recipe["name"]
        rating = [int(x) for x in recipe['rating'].split(',')
                  if x.strip().isdigit()]
        recipe_dictionary[name] = {"id": recipe["id"], "minutes": int(recipe["minutes"]), "ingredients": clean(
            recipe["ingredients"]), "tags": clean(recipe["tags"]), "rating": rating}
    return recipe_dictionary


def inverted_index(recipe_dictionary):
    # Returns a dictionary mapping ingredient names to a list of recipes that contain that ingredient
    inverted_idx = {}
    for name, info in recipe_dictionary.items():
        for ingredient in info["ingredients"]:
            ingredient = stemmer.stem(ingredient)
            if ingredient in inverted_idx:
                inverted_idx[ingredient].append(name)
            else:
                inverted_idx[ingredient] = [name]
    return inverted_idx


def allergy_inverted_index(recipe_dictionary):
    # Returns a dictionary mapping allergens to a list of recipes that contain that allergen
    inverted_idx = {}
    global allergies
    for i in allergies:
        inverted_idx[i] = []
    for name, info in recipe_dictionary.items():
        for al in allergies:
            for ingr in info["ingredients"]:
                if ingr in allergies[al]:
                    inverted_idx[al].append(name)
    return inverted_idx


# cleans the list inputted


def clean(l):
    ingr_list = re.findall(r"[\w -][\w -]+", l)
    new_list = []
    for ing in ingr_list:
        if ing[0] == " ":
            new_list.append(ing[1:])
        else:
            new_list.append(ing)
    return new_list

# code adapted from demo from class
# returns a list containing all ingredients in either of the lists


def merge_postings(postings1, postings2):
    merged_posting = []
    i, j = 0, 0
    postings1 = sorted(postings1)
    postings2 = sorted(postings2)
    while i < len(postings1) and j < len(postings2):
        if postings1[i] == postings2[j]:
            merged_posting.append(postings1[i])
            i += 1
            j += 1
        elif postings1[i] < postings2[j]:
            i += 1
        else:
            j += 1
    return merged_posting


# dish_list is list of strings, allergen is list of strings
# returns a list containing all ingredients in dish_list that does not contain allergen
def not_merge_postings(dish_list, allergen):
    # dish_list = [stemmer.stem(w.lower()) for w in dish_list]
    merged = merge_postings(dish_list, allergen)
    new_list = dish_list.copy()
    for t in merged:
        new_list.remove(t)
    return new_list

# calculates the jaccard similarity between two ingredient lists


def jaccard(ingr_list1, ingr_list2):
    set1 = set([stemmer.stem(w.lower()) for w in ingr_list1])
    set2 = set([stemmer.stem(w.lower()) for w in ingr_list2])
    if len(set.union(set1, set2)) == 0:
        return 0
    return len(set.intersection(set1, set2))/len(set.union(set1, set2))


def get_rating(dict, name):
    avg = np.mean(dict[name]["rating"])
    return avg


def preprocessing(ingredients, optional, restrictions, category, time):
    global mapping
    global ii
    global aii
    global time_lookup
    global category_lookup
    if mapping is None:
        query_sql = f"""SELECT * FROM rep limit 1000"""
        keys = ["id", "rating", "name", "minutes", "tags", "ingredients"]
        data = mysql_engine.query_selector(query_sql)
        zipping = [dict(zip(keys, i)) for i in data]
        mapping = preprocess(zipping)
        ii = inverted_index(mapping)
        aii = allergy_inverted_index(mapping)

    output = []
    if len(ingredients) ==0 and len(optional) == 0:
        output.append({"title": "No proper ingredient given."})
        return json.dumps(output)
    for i in ingredients:
        if i not in ii.keys():
            output.append({"title": "Ingredient '" + i + "' not found"})
            return json.dumps(output)
    ranked = search_rank(ingredients, optional, restrictions,
                         category, time, aii, ii, mapping, time_lookup, category_lookup)
    # print(ml_output_dict)
    for rep in ranked:
        name = rep[0]
        d = {"title": name, "descr": mapping[name]
             ["ingredients"], "link": URL + str(mapping[name]["id"]),
             "rating": np.round(get_rating(mapping, name), 1)}

        re_id = int(mapping[name]["id"])
        if re_id in ml_output_dict.keys():
            d['relevant_topic'] = ml_output_dict[re_id]
        else:
            d['relevant_topic'] = "Classic that everyone must try!"

        output.append(d)
    if len(output) == 0:
        output.append({"title": "No recipe found."})
    return json.dumps(output)

@app.route("/")
def home():
    return render_template('base.html', title="sample html")


def clean_ingredient(str):
    cleaned = re.findall(r"[a-zA-Z]+", str)
    return cleaned


@app.route("/recipes")
def recipe_search():
    # need to first validate ingredients
    # destem ingredients (to deal with plural ingredients)
    ingredients = request.args.get("mandatory")
    ingr = ingredients.split(",")
    # cleaning ingredients
    cleaned_ingr = []
    for i in ingr:
        for cleaned in clean_ingredient(i):
            cleaned_ingr.append(stemmer.stem(cleaned.lower()))

    # removing duplicates
    dupe = set(cleaned_ingr)
    no_dupe_ingr = list(dupe)

    # clean optional ingredients
    optional_ingredients = request.args.get("optional")
    optional = optional_ingredients.split(",")
    cleaned_optional = []
    for i in optional:
        for cleaned in clean_ingredient(i):
            cleaned_optional.append(stemmer.stem(cleaned.lower()))

    no_dupe_optional = list(set(cleaned_optional))

    restrictions = request.args.get("restrictions")
    restrict = restrictions.split(",")
    category = request.args.get("category")
    time = request.args.get("time")
    return preprocessing(no_dupe_ingr, no_dupe_optional, restrict, category, time)


# app.run(debug=True)
