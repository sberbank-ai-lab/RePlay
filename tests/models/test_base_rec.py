# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
from typing import Optional

import pytest
from pandas import DataFrame

from replay.models import Recommender
from tests.utils import spark, log


# pylint: disable=missing-class-docstring, too-many-arguments
class DerivedRec(Recommender):
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        pass

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        pass


@pytest.fixture
def model():
    return DerivedRec()


@pytest.mark.parametrize("array", [None, [1, 2, 2, 3]])
def test_extract_if_needed(spark, model, array):
    log = spark.createDataFrame(data=[[1], [2], [3]], schema=["test"])
    assert sorted(
        list(model._get_ids(array or log, "test").toPandas()["test"])
    ) == [1, 2, 3]


def test_users_count(model, log):
    with pytest.raises(AttributeError):
        model.max_user()
    model.fit(log)
    assert model.max_user == 3


def test_items_count(model, log):
    with pytest.raises(AttributeError):
        model.max_item()
    model.fit(log)
    assert model.max_item == 3


def test_str(model):
    assert str(model) == "DerivedRec"
