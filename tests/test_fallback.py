# pylint: disable-all
import pandas as pd

from replay.models import KNN
from replay.scenarios import Fallback
from replay.utils import fallback, convert2spark
from tests.utils import log, log2, spark


def test_fallback():
    base = pd.DataFrame({"user_idx": [1], "item_idx": [1], "relevance": [1]})
    extra = pd.DataFrame(
        {"user_idx": [1, 1, 2], "item_idx": [1, 2, 1], "relevance": [1, 2, 1]}
    )
    base = convert2spark(base)
    extra = convert2spark(extra)
    res = fallback(base, extra, 2).toPandas()
    assert len(res) == 3
    assert res.user_idx.nunique() == 2
    a = res.loc[
        (res["user_idx"] == 1) & (res["item_idx"] == 1), "relevance"
    ].iloc[0]
    b = res.loc[
        (res["user_idx"] == 1) & (res["item_idx"] == 2), "relevance"
    ].iloc[0]
    assert a > b


def test_class(log, log2):
    model = Fallback(KNN(), threshold=3)
    s = str(model)
    assert s == "Fallback(KNN, PopRec)"
    model.fit(log2)
    (p1, p2), p3 = model.optimize(log, log2, k=1, budget=1)
    assert p2 is None
    assert p3 is None
    assert isinstance(p1, dict)
    model.predict(log2, k=1)
