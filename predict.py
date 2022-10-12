import pickle


def wine_review(features):
    pickled_model = pickle.load(open('WineReviews.pkl', 'rb'))
    review = str(round(list(pickled_model.predict([features]))[0]))

    return str("wine point " + review)
test_features=[42.0, 84373.0, 28343.0, 13.0, 51.0, 146.0, 0.0, 69.0, 6422.0]
wine_review(test_features)