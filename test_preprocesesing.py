from recipe_recommender import preprocess_data

def test_preprocess_data():
    sample_data = [
        ['Apple', 'Banana'],
        ['Banana'],
        ['Apple', 'Orange']
    ]
    expected_result = [
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 1]
    ]
    processed_data = preprocess_data(sample_data)
    assert processed_data == expected_result
