from jel.biencoder.predictor import predictors_loader

def test_predictors_loader():
    mention_predictor, entity_predictor = predictors_loader()

    # currently, only chive mention encoder is supported.
    assert len(mention_predictor.predict_json({'anchor_sent': '今日は<a>品川</a>に行った。'}
                                              )['contextualized_mention']) == 300
    assert len(entity_predictor.predict_json({"gold_title":"隅田川",
                                              "gold_ent_desc":"花火がよく上がる"}
                                             )['contextualized_entity']) == 300

