# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import numpy as np
import pandas as pd

from pyspark.sql import functions as sf
from statsmodels.stats.proportion import proportion_confint

from replay.models import RandomRec
from replay.utils import convert2spark
from tests.utils import log, spark, sparkDataFrameEqual, sparkDataFrameNonEqual


@pytest.fixture
def uniform_seed():
    model = RandomRec(seed=123)
    return model


@pytest.fixture
def uniform_no_seed():
    model = RandomRec()
    return model


@pytest.fixture
def popular_based_seed():
    model = RandomRec(distribution="popular_based", seed=123)
    return model


def test_works(log, uniform_seed, uniform_no_seed, popular_based_seed):
    for model in [uniform_seed, uniform_no_seed, popular_based_seed]:
        model.fit(log)
        model.item_popularity.count()
        model.item_popularity.show()


def test_popularity_matrix(
    log, uniform_seed, uniform_no_seed, popular_based_seed
):
    uniform_matrix = (
        log.select("item_id").distinct().withColumn("probability", sf.lit(1.0))
    )
    popular_based_matrix = log.groupby("item_id").agg(
        sf.countDistinct("user_id").astype("double").alias("probability")
    )
    for model in [uniform_seed, uniform_no_seed]:
        model.fit(log)
        sparkDataFrameEqual(
            model._convert_back(
                model.item_popularity,
                log.schema["user_id"].dataType,
                log.schema["item_id"].dataType,
            ),
            uniform_matrix,
        )

    popular_based_seed.fit(log)
    sparkDataFrameEqual(
        popular_based_seed._convert_back(
            popular_based_seed.item_popularity,
            log.schema["user_id"].dataType,
            log.schema["item_id"].dataType,
        ),
        popular_based_matrix,
    )


def test_predict(log, uniform_seed, uniform_no_seed, popular_based_seed):
    # fixed seed provides reproducibility and diversity
    for model in [uniform_seed, popular_based_seed]:
        model.fit(log)
        pred = model.predict(log, k=1)
        model.fit(log)
        pred_same = model.predict(log, k=1)
        # reproducibility of predictions after model refit for fixed seed
        sparkDataFrameEqual(pred, pred_same)
        # diversity in fitted model's predictions
        pred_new = model.predict(log, k=1)
        sparkDataFrameNonEqual(pred_same, pred_new)

    # no seed provides diversity
    uniform_no_seed.fit(log)
    pred = model.predict(log, k=1)
    uniform_no_seed.fit(log)
    pred_same = model.predict(log, k=1)
    sparkDataFrameNonEqual(pred, pred_same)
    pred_new = model.predict(log, k=1)
    sparkDataFrameNonEqual(pred_same, pred_new)
