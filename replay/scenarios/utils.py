import logging
from typing import Optional
import numpy as np

from pyspark.sql import DataFrame

from replay.constants import AnyDataFrame
from replay.models import PopRec, ALSWrap
from replay.scenarios.two_stages_scenario import TwoStagesScenario
from replay.scenarios.fallback import Fallback
from replay.session_handler import State
from replay.splitters import DateSplitter, UserSplitter


def choose_splitter(data: DataFrame):
    """Автовыбор сплиттера по данным"""
    if "timestamp" in data.columns:
        splitter = DateSplitter(0.2)
    else:
        splitter = UserSplitter(item_test_size=0.2, shuffle=True)
    return splitter


# pylint: disable=unused-argument
def choose_scenario(
    data: AnyDataFrame,
    user_features: Optional[AnyDataFrame],
    item_features: Optional[AnyDataFrame],
):
    """Автовыбор сценария по данным"""
    if user_features or item_features:
        scenario = TwoStagesScenario
    else:
        scenario = Fallback
    return scenario


def choose_candidates(train, test, candidates, k, criterion):
    """
    Отобрать модели, которые смогли побить бейзлайн
    :param data: данные для обучения
    :param candidates: список моделей-кандидатов
    :param k: длина предсказаний
    :param criterion: метрика для сравнения
    :return: список моделей, прошедших проверку
    """
    logger = logging.getLogger("replay")
    logger.info("Choosing candidates...")
    baseline = PopRec().fit_predict(train, k)
    base_value = criterion(baseline, test, k)
    logger.debug("PopRec baseline has value %f", base_value)
    res = []
    num_candidates = len(candidates)
    for num, model in enumerate(candidates):
        logger.debug(
            "Calculating default performance value for candidate model %d/%d",
            num,
            num_candidates,
        )
        pred = model.fit_predict(train, k)
        candidate_value = criterion(pred, test, k)
        logger.debug("Performance value is %f", candidate_value)
        if candidate_value >= base_value:
            res.append(model)
    if len(res) == 0:
        State().logger.warning("Ни одна модель не смогла побить попрек")
        res = [ALSWrap()]
    return res


# pylint: disable=too-many-arguments
def get_best_model(
    train,
    test,
    user_features,
    item_features,
    candidates,
    timeout,
    k,
    criterion,
):
    """
    Выбрать лучшую модель из кандидатов

    :param train: данные для обучения
    :param test: данные для проверки
    :param user_features: фичи пользователей
    :param item_features: фичи объектов
    :param candidates: список моделей-кандидатов
    :param timeout: время на оптимизацию всех моделей
    :param k: длина списка рекомендаций
    :param criterion: метрика, по которой будет выбрана лучшая модель
    :return: лучшая модель
    """
    logger = logging.getLogger("replay")
    model_time = timeout // len(candidates)
    if model_time == 0:
        model_time = 1
    performance = []
    num_candidates = len(candidates)
    for num, model in enumerate(candidates):
        logger.debug(
            "Optimizing candidate models %d/%d...", num, num_candidates
        )
        model.optimize(
            train,
            test,
            user_features,
            item_features,
            criterion=criterion,
            k=k,
            budget=None,
            timeout=model_time,
        )
        performance.append(model.study.best_value)
    best_index = np.argmax(performance)
    return candidates[best_index]
