from catnet.eval.growth_aware import GrowthAwareEvaluator, GrowthEvalCase
from catnet.eval.metrics import cluster_purity, merge_error_rate, split_error_rate


def test_growth_aware_evaluator_reports_open_set_and_time_gap_metrics() -> None:
    cases = [
        GrowthEvalCase(
            query_obs_id="q1",
            true_cat_id="cat_a",
            candidates=[("cat_a", 0.95), ("cat_b", 0.2)],
            is_known=True,
            time_gap_days=3,
            age_bin="adult",
            site_id="yard",
            camera_id="front",
            view="front",
        ),
        GrowthEvalCase(
            query_obs_id="q2",
            true_cat_id="cat_b",
            candidates=[("cat_c", 0.91), ("cat_b", 0.88)],
            is_known=True,
            time_gap_days=220,
            age_bin="juvenile",
            site_id="yard",
            camera_id="side",
            view="left",
        ),
        GrowthEvalCase(
            query_obs_id="q3",
            true_cat_id=None,
            candidates=[("cat_a", 0.4)],
            is_known=False,
            time_gap_days=None,
            site_id="porch",
            camera_id="front",
            view="front",
        ),
    ]

    report = GrowthAwareEvaluator(threshold=0.86).evaluate(cases)

    assert report["overall"]["rank1"] == 0.5
    assert report["overall"]["BAUS"] == 1.0
    assert report["identity_persistence_score"] == 0.5
    assert report["long_term_drift_metric"] == 1.0
    assert any(bucket["bin"] == "180-365d" for bucket in report["time_gap_curve"])
    assert report["site"]["yard"]["rank1"] == 0.5


def test_graph_metrics_detect_purity_merge_and_split_errors() -> None:
    assignments = [
        ("cat_a", "cluster_1"),
        ("cat_a", "cluster_1"),
        ("cat_b", "cluster_1"),
        ("cat_b", "cluster_2"),
    ]

    assert cluster_purity(assignments) == 0.75
    assert merge_error_rate([("cat_a", "cat_a", True), ("cat_a", "cat_b", False)]) == 0.5
    assert split_error_rate(assignments) == 0.5
