def get_unrated_movie_id(user_id, rating_data):
    unique_product_id=set(rating_data['item_id'])
    rated_movie_id=set(rating_data.loc[rating_data['user_id']==user_id]['item_id'])
    unrated_movie_id=unique_product_id.difference(rated_movie_id)
    return unrated_movie_id