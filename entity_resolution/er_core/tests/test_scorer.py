from src.evaluate.scorer import entity_f05, macro_f05, score_detailed


def test_statement_example():
    # pred has one extra false positive: P=2/3, R=1 -> 0.714
    score = entity_f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert abs(score - 0.7142857) < 1e-5


def test_singleton_conventions():
    assert entity_f05([], []) == 1.0
    assert entity_f05(["S2-1"], []) == 0.0
    assert entity_f05([], ["S2-1"]) == 0.0


def test_perfect_and_macro():
    preds = {"S1-1": ["S2-1", "S3-1"], "S1-2": []}
    truth = {"S1-1": ["S2-1", "S3-1"], "S1-2": []}
    assert macro_f05(preds, truth) == 1.0

    preds = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-9"]}
    truth = {"S1-1": ["S2-1", "S3-1"], "S1-2": []}
    # entity 1 = 1.0, entity 2 (false merge on singleton) = 0.0
    assert macro_f05(preds, truth) == 0.5


def test_score_detailed_slices():
    preds = {"S1-1": ["S2-1"], "S1-2": [], "S1-3": ["S2-3"]}
    truth = {"S1-1": ["S2-1"], "S1-2": [], "S1-3": ["S2-3", "S3-3"]}
    detail = score_detailed(preds, truth)
    assert detail["macro_f05"] > 0.0
    assert detail["slices"]["n_singletons"] == 1
    assert detail["slices"]["singleton"] == 1.0
