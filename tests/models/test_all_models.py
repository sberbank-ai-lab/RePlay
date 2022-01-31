# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
from datetime import datetime

import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import (
    ALSWrap,
    ADMMSLIM,
    KNN,
    LightFMWrap,
    NeuroMF,
    PopRec,
    SLIM,
    MultVAE,
    Word2VecRec,
)

from tests.utils import spark, log, sparkDataFrameEqual

SEED = 123


@pytest.fixture
def log_to_pred(spark):
    return spark.createDataFrame(
        data=[
            [0, 2, datetime(2019, 9, 12), 3.0],
            [0, 4, datetime(2019, 9, 13), 2.0],
            [1, 5, datetime(2019, 9, 14), 4.0],
            [4, 0, datetime(2019, 9, 15), 3.0],
            [4, 1, datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(seed=SEED),
        ADMMSLIM(seed=SEED),
        KNN(),
        LightFMWrap(random_state=SEED),
        MultVAE(),
        NeuroMF(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
        PopRec(),
    ],
    ids=[
        "als",
        "admm_slim",
        "knn",
        "lightfm",
        "multvae",
        "neuromf",
        "slim",
        "word2vec",
        "poprec",
    ],
)
def test_predict_pairs_warm_only(log, log_to_pred, model):
    model.fit(log)
    recs = model.predict(
        log.unionByName(log_to_pred),
        k=3,
        users=log_to_pred.select("user_idx").distinct(),
        items=log_to_pred.select("item_idx").distinct(),
        filter_seen_items=False,
    )

    pairs_pred = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        log=log.unionByName(log_to_pred),
    )

    condition = ~sf.col("item_idx").isin([4, 5])
    if not model.can_predict_cold_users:
        condition = condition & (sf.col("user_idx") != 4)

    sparkDataFrameEqual(
        pairs_pred.select("user_idx", "item_idx"),
        log_to_pred.filter(condition).select("user_idx", "item_idx"),
    )

    recs_joined = (
        pairs_pred.withColumnRenamed("relevance", "pairs_relevance")
        .join(recs, on=["user_idx", "item_idx"], how="left")
        .sort("user_idx", "item_idx")
    )

    assert np.allclose(
        recs_joined.select("relevance").toPandas().to_numpy(),
        recs_joined.select("pairs_relevance").toPandas().to_numpy(),
    )


@pytest.mark.parametrize(
    "model",
    [
        ADMMSLIM(seed=SEED),
        KNN(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
    ],
    ids=["admm_slim", "knn", "slim", "word2vec",],
)
def test_predict_pairs_raises(log, model):
    with pytest.raises(ValueError, match="log is not provided,.*"):
        model.fit(log)
        model.predict_pairs(log.select("user_idx", "item_idx"))


def test_predict_pairs_raises_pairs_format(log):
    model = ALSWrap(seed=SEED)
    with pytest.raises(ValueError, match="pairs must be a dataframe with .*"):
        model.fit(log)
        model.predict_pairs(log, log)


# for NeighbourRec and ItemVectorModel
@pytest.mark.parametrize(
    "model, metric",
    [
        (ALSWrap(seed=SEED), "euclidean_distance_sim"),
        (ALSWrap(seed=SEED), "dot_product"),
        (ALSWrap(seed=SEED), "cosine_similarity"),
        (Word2VecRec(seed=SEED, min_count=0), "cosine_similarity"),
        (ADMMSLIM(seed=SEED), None),
        (KNN(), None),
        (SLIM(seed=SEED), None),
    ],
    ids=[
        "als_euclidean",
        "als_dot",
        "als_cosine",
        "w2v_cosine",
        "admm_slim",
        "knn",
        "slim",
    ],
)
def test_get_nearest_items(log, model, metric):
    model.fit(log.filter(sf.col("item_idx") != 3))
    res = model.get_nearest_items(items=[0, 1], k=2, metric=metric)

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_idx"].values()) == {
        0,
        1,
    }

    res = model.get_nearest_items(items=[0, 1], k=1, metric=metric)
    assert res.count() == 2

    # filter neighbours
    res = model.get_nearest_items(
        items=[0, 1], k=4, metric=metric, candidates=[0, 3],
    )
    assert res.count() == 1
    assert (
        len(
            set(res.toPandas().to_dict()["item_idx"].values()).difference(
                {0, 1}
            )
        )
        == 0
    )


def test_nearest_items_raises(log):
    model = PopRec()
    model.fit(log.filter(sf.col("item_idx") != 3))
    with pytest.raises(
        ValueError, match=r"Distance metric is required to get nearest items.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=None)

    with pytest.raises(
        ValueError,
        match=r"Use models with attribute 'can_predict_item_to_item' set to True.*",
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric="cosine_similarity")

        with pytest.raises(
            ValueError,
            match=r"Use models with attribute 'can_predict_item_to_item' set to True.*",
        ):
            model.get_nearest_items(
                items=[0, 1], k=2, metric="cosine_similarity"
            )


def test_filter_seen(log):
    model = PopRec()
    # filter seen works with empty log to filter (cold_user)
    model.fit(log.filter(sf.col("user_idx") != 0))
    pred = model.predict(log=log, users=[3], k=5)
    assert pred.count() == 2

    # filter seen works with log not presented during training (for user1)
    pred = model.predict(log=log, users=[0], k=5)
    assert pred.count() == 1

    # filter seen turns off
    pred = model.predict(log=log, users=[0], k=5, filter_seen_items=False)
    assert pred.count() == 4
