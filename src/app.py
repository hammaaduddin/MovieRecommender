from flask import Flask, render_template, request, jsonify
import pandas as pd
from recommenders import recommendByTitle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies.csv")
movies["doc"] = movies["title"]

tfidf = TfidfVectorizer(stop_words="english")
tfidfMatrix = tfidf.fit_transform(movies["doc"])
cosineSim = cosine_similarity(tfidfMatrix, tfidfMatrix)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommendByTitle(title, top_n=5):
    if title not in indices:
        return []
    idx = indices[title]
    simScores = list(enumerate(cosineSim[idx]))
    simScores = sorted(simScores, key= lambda x:x[1], reverse=True)[1:top_n+1]
    movieIndicies = [i[0] for i in simScores]
    return movies["title"].iloc[movieIndicies].tolist()

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])

def index():
    recommendations = []
    query = ""
    if request.method == "POST":
        query = request.form["title"]
        recommendations = recommendByTitle(query, top_n=5)
    return render_template("templates/index.html", query=query, recommendations=recommendations)

if __name__ ==  "__main__":
    app.run(debug=True) 


