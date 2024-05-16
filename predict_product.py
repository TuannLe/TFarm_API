import pandas as pd
from unratedmovie import get_unrated_movie_id
from pred_unrated import get_pred_unrated_item
import pickle

def predict_product():
    rating_data= pd.read_csv('df_electronics.csv')
    anime= pd.read_csv('product_df.csv')

    #replace -1 with 0 in rating_data
    rating_data['rating'] = rating_data['rating'].replace(-1,0)

    # Replace this with the desired user ID you want to make predictions for
    user_id = 142967

    unrated_movie_id = get_unrated_movie_id(user_id, rating_data=rating_data)

    # Ensure that the user has rated some anime before proceeding with predictions
    if unrated_movie_id:
        trained_model = pickle.load(open('knn_model_1.sav', 'rb'))

        # Get predicted unrated anime using the trained model
        predicted_unrated_anime_df = get_pred_unrated_item(user_id=user_id, estimator=trained_model, unrated_movie_id=unrated_movie_id)

        k = 6
        top_anime_svd = predicted_unrated_anime_df.head(k).copy()

        # Create a dictionary mapping anime_id to their respective names and genres
        anime_id_to_name = anime.set_index('item_id')['name'].to_dict()

        # Map the anime_id to their names and genres in the 'top_anime_svd' DataFrame
        top_anime_svd['name'] = top_anime_svd['item_id'].map(anime_id_to_name)

        print(top_anime_svd)
    else:
        print("User with ID {} has not rated any product.".format(user_id))

if __name__ == "__main__":
    predict_product()