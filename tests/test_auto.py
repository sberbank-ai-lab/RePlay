# pylint: disable-all

from replay.scenarios import autorec, Fallback
from tests.utils import ml1m


def test_auto(ml1m):
    model = autorec(ml1m, None, None, 4)
    assert isinstance(model, Fallback)
