import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = ""
MYSQL_PORT = 3306
MYSQL_DATABASE = "kardashiandb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this

# global variables
mapping = None
ii = None
aii = None

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


def search_rank(query, allergens, allergy_inverted_index, inverted_index, recipe_dict):
    # query is a list of strings, allergens is a list of strings, inverted_index is
    # a dictionary mapping ingredients to which recipes they appear in. recipe_dict
    # is a dictionary mapping recipe names to id, ingredients, and tags.
    # Returns a ranked list of the top 20 recipes
    postings1 = inverted_index[query[0]]
    for ingredient in query:
        postings2 = inverted_index[ingredient]
        postings1 = merge_postings(postings1, postings2)

    for allergen in allergens:
        if allergen == "":
            break
        allergen_postings = allergy_inverted_index[allergen]
        postings1 = not_merge_postings(postings1, allergen_postings)

    similarity_ranking = []
    for posting in postings1:
        ingredients = recipe_dict[posting]['ingredients']
        jaccard_sim = jaccard(query, ingredients)
        similarity_ranking.append((posting, jaccard_sim))

    similarity_ranking.sort(reverse=True, key=lambda x: x[1])
    boundary = min(20, len(similarity_ranking))
    return similarity_ranking[:boundary]


def preprocess(recipe_list):
    # Returns a dictionary mapping recipe name to id, a list of ingredients, and a list of tags
    # Would be good to clean the recipe name
    recipe_dictionary = {}
    for recipe in recipe_list:
        name = recipe["name"]
        recipe_dictionary[name] = {"id": recipe["id"], "ingredients": clean(
            recipe["ingredients"]), "tags": clean(recipe["tags"])}
    return recipe_dictionary


def inverted_index(recipe_dictionary):
    # Returns a dictionary mapping ingredient names to a list of recipes that contain that ingredient
    inverted_idx = {}
    for name, info in recipe_dictionary.items():
        for ingredient in info["ingredients"]:
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
    merged = merge_postings(dish_list, allergen)
    new_list = dish_list.copy()
    for t in merged:
        new_list.remove(t)
    return new_list

# calculates the jaccard similarity between two ingredient lists


def jaccard(ingr_list1, ingr_list2):
    set1 = set(ingr_list1)
    set2 = set(ingr_list2)
    if len(set.union(set1, set2)) == 0:
        return 0
    return len(set.intersection(set1, set2))/len(set.union(set1, set2))


def preprocessing(ingredients, restrictions, category, time):
    # currently search does not use category or time
    global mapping
    global ii
    global aii
    if mapping is None:
        query_sql = f"""SELECT * FROM rep2"""
        keys = ["name", "id", "minutes", "tags", "ingredients"]
        data = mysql_engine.query_selector(query_sql)
        zipping = [dict(zip(keys, i)) for i in data]
        mapping = preprocess(zipping)
        ii = inverted_index(mapping)
        aii = allergy_inverted_index(mapping)
    ranked = search_rank(ingredients, restrictions, aii, ii, mapping)
    output = []
    for rep in ranked:
        name = rep[0]
        d = {"title": name, "descr": mapping[name]["ingredients"]}
        output.append(d)
    return json.dumps(output)


def edit_dist(query, ingredient):
    "Calculate the minimum edit distance between the query and an existing ingredient"
    n_rows = len(query) + 1
    n_cols = len(ingredient) + 1
    # construct an edit distance matrix, assume instertion cost = 1,
    # deletion cost = 1, and substitution cost = 2
    ins_cost = 1
    del_cost = 1
    mat = np.zeros((n_rows, n_cols))
    mat[0] = np.arange(n_cols)
    mat[:, 0] = np.arange(n_rows)
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            sub_cost = 0 if query[i-1] == ingredient[j-1] else 2
            mat[i, j] = min(mat[i-1, j] + del_cost,
                            mat[i-1, j-1] + sub_cost, mat[i, j-1] + ins_cost)
    return mat[len(query), len(ingredient)]


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


def clean_ingredient(str):
    cleaned = re.findall(r"[a-zA-Z ]+", str)
    return cleaned


@app.route("/recipes")
def recipe_search():
    # need to first validate ingredients
    # destem ingredients (to deal with plural ingredients)
    ingredients = request.args.get("ingredients")
    ingr = ingredients.split(",")
    cleaned_ingr = []
    for i in ingr:
        for cleaned in clean_ingredient(i):
            cleaned_ingr.append(cleaned.lower())
    restrictions = request.args.get("restrictions")
    restrict = restrictions.split(",")
    category = request.args.get("category")
    time = request.args.get("time")
    return preprocessing(cleaned_ingr, restrict, category, time)


# app.run(debug=True)
