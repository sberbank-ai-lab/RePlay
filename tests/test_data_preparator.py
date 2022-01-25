# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, too-many-arguments, invalid-name
from datetime import datetime
from copy import deepcopy
from unittest.mock import Mock

import pytest
import pandas as pd
from pyspark.sql import functions as sf, DataFrame
from pyspark.sql.types import StringType, StructType, IntegerType

from replay.constants import LOG_SCHEMA
from replay.data_preparator import (
    DataPreparator,
    CatFeaturesTransformer,
    Indexer,
)
from replay.utils import convert2spark
from tests.utils import (
    item_features,
    long_log_with_features,
    short_log_with_features,
    spark,
    sparkDataFrameEqual,
)


def test_preparator():
    p = DataPreparator()
    in_df = pd.DataFrame({"user": [4], "item_id": [3]})
    out_df = pd.DataFrame({"user_idx": [0], "item_idx": [0]})
    out_df = convert2spark(out_df)
    out_df = out_df.withColumn(
        "user_idx", sf.col("user_idx").cast(IntegerType())
    )
    out_df = out_df.withColumn(
        "item_idx", sf.col("item_idx").cast(IntegerType())
    )
    res, _, _ = p(in_df, mapping={"user_id": "user"})
    assert isinstance(res, DataFrame)
    assert set(res.columns) == {"user_idx", "item_idx"}
    sparkDataFrameEqual(res, out_df)
    res = p.back(res)
    assert set(res.columns) == {"user_id", "item_id"}


@pytest.fixture
def indexer():
    return Indexer()


def test_indexer(indexer, long_log_with_features):
    indexer.fit(long_log_with_features, long_log_with_features)
    res = indexer.transform(long_log_with_features)
    log = indexer.inverse_transform(res)
    sparkDataFrameEqual(log, long_log_with_features)


# categorical features transformer tests
def get_transformed_features(transformer, train, test):
    transformer.fit(train)
    return transformer.transform(test)


def test_cat_features_transformer(item_features):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer(cat_cols_list=["class"]),
        train=item_features.filter(sf.col("class") != "dog"),
        test=item_features,
    )
    assert "class" not in transformed.columns
    assert "iq" in transformed.columns and "color" in transformed.columns
    assert (
        "ohe_class_dog" not in transformed.columns
        and "ohe_class_cat" in transformed.columns
    )
    assert (
        transformed.filter(sf.col("item_id") == "i6")
        .select("ohe_class_mouse")
        .collect()[0][0]
        == 1.0
    )


def test_cat_features_transformer_date(
    long_log_with_features, short_log_with_features,
):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer(["timestamp"]),
        train=long_log_with_features,
        test=short_log_with_features,
    )
    assert (
        "ohe_timestamp_20190101000000" in transformed.columns
        and "item_id" in transformed.columns
    )


def test_cat_features_transformer_empty_list(
    long_log_with_features, short_log_with_features,
):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer([]),
        train=long_log_with_features,
        test=short_log_with_features,
    )
    assert len(transformed.columns) == 4
    assert "timestamp" in transformed.columns
