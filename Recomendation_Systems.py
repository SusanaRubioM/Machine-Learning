import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#Data
ratings_dict = {
    "user_id": [1, 2, 3, 4, 5],
    "movie_id": [101, 101, 102, 103, 104],
    "rating": [5, 4, 5, 3, 2]
}

ratings = pd.DataFrame(ratings_dict)
#utility matrix: represents qualifies for each item
utility_matrix = ratings.pivot_table(values='rating', index='user_id', columns='movie_id', fill_value=0)
print(utility_matrix)

#cosine similarity between items
item_similarity = cosine_similarity(utility_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=utility_matrix.columns, columns=utility_matrix.columns)
print(item_similarity_df)

"""
user_id(int), 
utility_matrix(matrix), 
item_similarity_df(panda dataFrame), 
n_recommendations(int)
"""
def recommend_movies(user_id, utility_matrix, item_similarity_df, n_recommendations=3):

    user_ratings = utility_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index.tolist()

    scores = item_similarity_df[rated_movies].sum(axis=1) / len(rated_movies)
    scores = scores.drop(rated_movies)  # Eliminar las películas que el usuario ya ha visto

    recommendations = scores.nlargest(n_recommendations).index.tolist()
    return recommendations


recommendations = recommend_movies(1, utility_matrix, item_similarity_df, n_recommendations=3)
print(f"Recomendaciones para el usuario: {recommendations}")
