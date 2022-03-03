# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
from pyspark.sql import functions as sf

from replay.models import ALSWrap, KNN, PopRec, LightFMWrap
from replay.scenarios.two_stages.models_ensemble import RecEnsemble

from tests.utils import (
    spark,
    sparkDataFrameEqual,
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
)


def fitted_ensemble(long_log_with_features, user_features, item_features):
    ensemble = RecEnsemble(
        models=[
            ALSWrap(rank=4),
            KNN(num_neighbours=4),
            LightFMWrap(no_components=4),
        ],
        fallback_model=PopRec(),
    )

    ensemble.fit(
        long_log_with_features.filter(sf.col("user_idx") < 2),
        user_features,
        item_features.filter(sf.col("iq") > 4),
    )

    return ensemble


def test_fit_predict(
    long_log_with_features,
    user_features,
    item_features,
):
    ensemble = fitted_ensemble(
        long_log_with_features, user_features, item_features
    )
    num_recs_by_one_model = [1, 2, 3]
    # with fallback model
    pred = ensemble.predict(
        long_log_with_features.filter(sf.col("user_idx") < 2),
        users=long_log_with_features.select("user_idx").distinct(),
        items=long_log_with_features.select("item_idx").distinct(),
        num_recs_by_one_model=num_recs_by_one_model,
        filter_seen_items=False,
    )
    assert sorted(pred.columns) == [
        "item_idx",
        "rel_0_ALSWrap",
        "rel_1_KNN",
        "rel_2_LightFMWrap",
        "rel_fb",
        "user_idx",
    ]
    assert pred.count() == long_log_with_features.select(
        "user_idx"
    ).distinct().count() * sum(num_recs_by_one_model)

    # no fallback model
    ensemble.fallback_model = None
    pred = ensemble.predict(
        long_log_with_features.filter(sf.col("user_idx") < 2),
        users=long_log_with_features.select("user_idx").distinct(),
        items=long_log_with_features.select("item_idx").distinct(),
        num_recs_by_one_model=num_recs_by_one_model,
        filter_seen_items=True,
    )
    assert sorted(pred.columns) == [
        "item_idx",
        "rel_0_ALSWrap",
        "rel_1_KNN",
        "rel_2_LightFMWrap",
        "user_idx",
    ]
    assert pred.count() < long_log_with_features.filter(
        sf.col("user_idx") < 2
    ).select("user_idx").distinct().count() * sum(num_recs_by_one_model)
    assert pred.select("user_idx").distinct().count() == 2


def test_add_relevance_vectors(
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
):
    ensemble = fitted_ensemble(
        long_log_with_features, user_features, item_features
    )
    res = ensemble.add_relevance_vectors(
        pairs=short_log_with_features.select("item_idx", "user_idx"),
        log=long_log_with_features,
        user_features=user_features,
        item_features=item_features,
        calc_relevance=True,
        calc_rank=True,
        add_features=True,
    )
    assert short_log_with_features.count() == res.count()
    assert "m_0_fm_0" in res.columns
    assert "m_2_item_bias" in res.columns


def test_optimize(
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
):
    ensemble = fitted_ensemble(
        long_log_with_features, user_features, item_features
    )
    param_borders = [{"rank": [1, 10]}, {}, {"no_components": [1, 10]}, None]
    # with fallback
    first_level_params, fallback_params = ensemble.optimize(
        train=long_log_with_features,
        test=short_log_with_features,
        user_features=user_features,
        item_features=item_features,
        param_borders=param_borders,
        k=1,
        budget=1,
    )
    assert len(first_level_params) == 3
    assert first_level_params[1] is None
    assert list(first_level_params[0].keys()) == ["rank"]
    assert fallback_params is None

    # no fallback works
    ensemble.fallback_model = None
    ensemble.optimize(
        train=long_log_with_features,
        test=short_log_with_features,
        user_features=user_features,
        item_features=item_features,
        param_borders=param_borders[:3],
        k=1,
        budget=1,
    )
