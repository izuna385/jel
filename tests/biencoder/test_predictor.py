from jel.biencoder.predictor import mention_predictor_loader

def test_mention_predictor_loader():
    predictor = predictor_loader()
    # currently, only chive mention encoder is supported.
    assert len(predictor.predict('今日は<a>品川</a>に行った。')['contextualized_mention']) == 300

