# pylint: disable-all
import pandas as pd
import pytest

from replay.metrics import NDCG
from replay.scenarios import autorec, Fallback, TwoStagesScenario
from replay.scenarios.utils import (
    choose_candidates,
    choose_splitter,
    choose_scenario,
)
from replay.splitters import DateSplitter, UserSplitter
from replay.models import PopRec, ALSWrap
from replay.utils import convert2spark
from tests.utils import ml1m


def test_auto(ml1m):
    model = autorec(ml1m, None, None, 4)
    assert isinstance(model, Fallback)


def test_choose_default(ml1m):
    splitter = DateSplitter(0.2)
    train, test = splitter.split(ml1m)
    candidates = [PopRec()]
    model = choose_candidates(train, test, candidates, 30, NDCG())
    assert isinstance(model, ALSWrap)


@pytest.mark.parametrize(
    "data,split_class",
    [({"timestamp": [1]}, DateSplitter), ({"other": [1]}, UserSplitter)],
)
def test_choose_splitter(data, split_class):
    df = pd.DataFrame(data)
    splitter = choose_splitter(convert2spark(df))
    assert isinstance(splitter, split_class)


@pytest.mark.parametrize(
    "data, user_features, item_features, answer",
    [
        (None, None, 1, TwoStagesScenario),
        (None, 1, None, TwoStagesScenario),
        (None, None, None, Fallback),
    ],
)
def test_choose_scenario(data, user_features, item_features, answer):
    scenario = choose_scenario(data, user_features, item_features)
    assert scenario is answer
