from typing import Optional

from replay.constants import AnyDataFrame
from replay.metrics import NDCG
from replay.models import KNN, ALSWrap, MultVAE, SLIM
from replay.scenarios.utils import (
    choose_splitter,
    choose_scenario,
    choose_candidates,
    get_best_model,
)
from replay.scenarios.basescenario import split_train_time
from replay.utils import convert2spark


def autorec(
    data: AnyDataFrame,
    user_features: Optional[AnyDataFrame] = None,
    item_features: Optional[AnyDataFrame] = None,
    timeout: Optional[int] = 10,
):
    """
    Посчитать лучшую модель в автоматическом режиме по предустановленным настройкам.

    :param data: данные, на которых необходимо обучить модель
    :param user_features: фичи пользователей
    :param item_features: фичи объектов
    :param timeout: время для оптимизации моделей. Не совпадает со временем на весь процесс.
    :return: сценарий с моделью, обученный на лучших параметрах
    """
    k = 30
    criterion = NDCG()
    data = convert2spark(data)
    splitter = choose_splitter(data)
    train, test = splitter.split(data)
    scenario = choose_scenario(data, user_features, item_features)

    candidates = [KNN(), ALSWrap(), MultVAE(), SLIM()]
    candidates = choose_candidates(train, test, candidates, k, criterion)

    choose_time, optimize_time = split_train_time(timeout)
    model = get_best_model(
        train,
        test,
        user_features,
        item_features,
        candidates,
        choose_time,
        k=k,
        criterion=criterion,
    )
    # pylint: disable=unexpected-keyword-arg
    scenario = scenario(main_model=model)
    scenario.optimize(
        train,
        test,
        criterion=criterion,
        k=k,
        budget=None,
        timeout=optimize_time,
    )
    scenario.fit(data, user_features, item_features)
    return scenario
