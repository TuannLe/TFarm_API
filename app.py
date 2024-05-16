import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from unratedmovie import get_unrated_movie_id
from pred_unrated import get_pred_unrated_item
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return{"message":"Hello World"}

class Recommender(Resource):
    def post(self):
        data = request.get_json()
        user_id = data['user_id']

        rating_data = pd.read_csv('df_electronics.csv')
        anime = pd.read_csv('product_df.csv')

        # replace -1 with 0 in rating_data
        rating_data['rating'] = rating_data['rating'].replace(-1, 0)

        # Replace this with the desired user ID you want to make predictions for


        unrated_movie_id = get_unrated_movie_id(user_id, rating_data=rating_data)

        # Ensure that the user has rated some anime before proceeding with predictions
        if unrated_movie_id:
            trained_model = pickle.load(open('knn_model_1.sav', 'rb'))

            # Get predicted unrated anime using the trained model
            predicted_unrated_anime_df = get_pred_unrated_item(user_id=user_id, estimator=trained_model,
                                                               unrated_movie_id=unrated_movie_id)

            k = 5
            top_anime_svd = predicted_unrated_anime_df.head(k).copy()

            # Create a dictionary mapping anime_id to their respective names and genres
            anime_id_to_name = anime.set_index('item_id')['name'].to_dict()

            # Map the anime_id to their names and genres in the 'top_anime_svd' DataFrame
            top_anime_svd['name'] = top_anime_svd['item_id'].map(anime_id_to_name)
            return {
                "user_id": top_anime_svd['user_id'].to_list(),
                "item_id": top_anime_svd['item_id'].to_list(),
                "predicted_rating": top_anime_svd['predicted_rating'].to_list(),
                "name": top_anime_svd['name'].to_list(),
            }
        else:
            return {"message": "User with ID {} has not rated any anime.".format(user_id)}




api.add_resource(HelloWorld, '/')
api.add_resource(Recommender, '/recommender')
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-port', type=str, default=5000, help='port to listen on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=True)