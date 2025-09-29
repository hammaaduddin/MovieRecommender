import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

mostPopular = ratings.groupby("movieId").size().sort_values(ascending=False)
top10 = mostPopular.head(10).index

#print(movies[movies["movieId"].isin(top10)])

movies["doc"] = movies["title"]

tfidf = TfidfVectorizer(stop_words="english")
tfidfMatrix = tfidf.fit_transform(movies["doc"])
cosineSim = cosine_similarity(tfidfMatrix, tfidfMatrix)

indices= pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommendByTitle(title, top_n=10):
    idx = indices[title]
    simScores = list(enumerate(cosineSim[idx]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movieIndices = [i[0] for i in simScores]
    return movies["title"].iloc[movieIndices].tolist()

#print(recommendByTitle("Toy Story (1995)"))

userMovie = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
itemSimilarity = cosine_similarity(userMovie.T)
movieIds = userMovie.columns.tolist()
movieIdToIdx = {m:i for i,m in enumerate(movieIds)}
idxToMovieId = {i:m for m,i in movieIdToIdx.items()}

def recommendSimilarMovie(movieId, top_n=10):
    i = movieIdToIdx[movieId]
    simScores = list(enumerate(itemSimilarity[i]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recIds = [idxToMovieId[s[0]] for s in simScores]
    return movies[movies["movieId"].isin(recIds)]["title"].to_list()

#print(recommendSimilarMovie(1))