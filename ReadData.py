import pandas as pd

ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["userId", "movieId", "rating", "timeestamp"])

ratings.to_csv("data/ratings.csv", index=False)

movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", names=["movieId", "title", "release_date", "viedo_release_date", "imdb_url"] + [f"genre_{i}" for i in range(19)])

movies = movies[["movieId", "title"]]
movies.to_csv("data/movies.csv", index=False)

print(ratings.head())
print(movies.head())