import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

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
def not_merge_postings(dish_list, allergen):
    merged = merge_postings(dish_list, allergen)
    new_list = dish_list.copy()
    for t in merged:
        new_list.remove(t)
    return new_list


def jaccard(ingr_list1, ingr_list2):
    set1 = set(ingr_list1)
    set2 = set(ingr_list2)
    print(set1)
    print(set1.intersection(set2))
    if len(set.union(set1, set2)) == 0:
        return 0
    return len(set.intersection(set1, set2))/len(set.union(set1, set2))


def get_sql_recipes():
    # query_sql = f"""SELECT name,id, tags, ingredients FROM episodes"""
    query_sql = f"""SELECT * FROM episodes"""
    keys = ["name", "id", "tags", "ingredients"]
    data = mysql_engine.query_selector(query_sql)
    print(data)
    return [dict(zip(keys, i)) for i in data]


def sql_search(episode):
    query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
    keys = ["id", "title", "descr"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys, i)) for i in data])


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return get_sql_recipes()


app.run(debug=True)
