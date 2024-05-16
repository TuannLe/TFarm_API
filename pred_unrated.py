import pandas as pd

def get_pred_unrated_item(user_id, estimator, unrated_movie_id):
    predicted_unrated_anime = {
        'user_id': user_id,
        'item_id': [],
        'predicted_rating': []
    }
    for id in unrated_movie_id:
        pred_id = estimator.predict(uid=user_id, iid=id)  # Use 'user_id' directly here
        # append
        predicted_unrated_anime['item_id'].append(id)
        predicted_unrated_anime['predicted_rating'].append(pred_id.est)
    predicted_unrated_anime_df = pd.DataFrame(predicted_unrated_anime)
    predicted_unrated_anime_df.sort_values(by='predicted_rating', ascending=False, inplace=True)
    return predicted_unrated_anime_df