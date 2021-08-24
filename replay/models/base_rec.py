# pylint: disable=too-many-lines
"""
Base abstract classes:
- BaseRecommender - the simplest base class
- Recommender - base class for models that fit on interaction log
- HybridRecommender - base class for models that accept user or item features
- UserRecommender - base class that accepts only user features, but not item features
- NeighbourRec - base class that requires log at prediction time
"""
import collections
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Union, Sequence, Tuple

import pandas as pd
from optuna import create_study
from optuna.samplers import TPESampler
from pyspark.ml.feature import IndexToString, StringIndexer, StringIndexerModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.column import Column

from replay.constants import AnyDataFrame
from replay.metrics import Metric, NDCG
from replay.optuna_objective import SplitData, MainObjective
from replay.session_handler import State
from replay.utils import get_top_k_recs, convert2spark


class BaseRecommender(ABC):
    """ Base recommender """

    model: Any
    user_indexer: StringIndexerModel
    item_indexer: StringIndexerModel
    inv_user_indexer: IndexToString
    inv_item_indexer: IndexToString
    _logger: Optional[logging.Logger] = None
    can_predict_cold_users: bool = False
    can_predict_cold_items: bool = False
    _search_space: Optional[
        Dict[str, Union[str, Sequence[Union[str, int, float]]]]
    ] = None
    _objective = MainObjective
    study = None

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Searches best parameters with optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
        :param param_grid: a dictionary with search grid, where
            key is the parameter name and value is the range of possible values``{param: [low, high]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :return: dictionary with best parameters
        """
        if self._search_space is None:
            self.logger.warning(
                "%s has no hyper parameters to optimize", str(self)
            )
            return None
        train = convert2spark(train)
        test = convert2spark(test)

        user_features_train, user_features_test = self._train_test_features(
            train, test, user_features, "user_id"
        )
        item_features_train, item_features_test = self._train_test_features(
            train, test, item_features, "item_id"
        )

        users = test.select("user_id").distinct()
        items = test.select("item_id").distinct()
        split_data = SplitData(
            train,
            test,
            users,
            items,
            user_features_train,
            user_features_test,
            item_features_train,
            item_features_test,
        )
        if param_grid is None:
            params = self._search_space.keys()
            vals = [None] * len(params)
            param_grid = dict(zip(params, vals))
        if self.study is None:
            self.study = create_study(
                direction="maximize", sampler=TPESampler()
            )
        objective = self._objective(
            search_space=param_grid,
            split_data=split_data,
            recommender=self,
            criterion=criterion,
            k=k,
        )
        self.study.optimize(objective, budget)
        return self.study.best_params

    @staticmethod
    def _train_test_features(train, test, features, column):
        if features is not None:
            features = convert2spark(features)
            features_train = features.join(
                train.select(column).distinct(), on=column
            )
            features_test = features.join(
                test.select(column).distinct(), on=column
            )
        else:
            features_train = None
            features_test = None
        return features_train, features_test

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Set model parameters

        :param params: dictionary param name - param value
        :return:
        """
        for param, value in params.items():
            setattr(self, param, value)
        self._clear_cache()

    def __str__(self):
        return type(self).__name__

    def _fit_wrap(
        self,
        log: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        force_reindex: bool = True,
    ) -> None:
        """
        Wrapper for fit to allow for fewer arguments in a model.

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features
            ``[user_id, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id, timestamp]`` + feature columns
        :param force_reindex: create indexers again, even if they were created previously
        :return:
        """
        self.logger.debug("Starting fit %s", type(self).__name__)
        log, user_features, item_features = [
            convert2spark(df) for df in [log, user_features, item_features]
        ]

        if "user_indexer" not in self.__dict__ or force_reindex:
            self.logger.debug("Creating indexers")
            self._create_indexers(log, user_features, item_features)
        self.logger.debug("Main fit stage")

        log, user_features, item_features = [
            self._convert_index(df)
            for df in [log, user_features, item_features]
        ]
        self._fit(log, user_features, item_features)

    def _create_indexers(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Creates indexers to map raw id to numerical idx so that spark can handle them.
        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features (must have ``user_id``)
        :param item_features: item features (must have ``item_id``)
        :return:
        """
        if user_features is None:
            users = log.select("user_id")
        else:
            users = log.select("user_id").union(
                user_features.select("user_id")
            )
        if item_features is None:
            items = log.select("item_id")
        else:
            items = log.select("item_id").union(
                item_features.select("item_id")
            )
        self.user_indexer = StringIndexer(
            inputCol="user_id", outputCol="user_idx"
        ).fit(users)
        self.item_indexer = StringIndexer(
            inputCol="item_id", outputCol="item_idx"
        ).fit(items)
        self.inv_user_indexer = IndexToString(
            inputCol="user_idx",
            outputCol="user_id",
            labels=self.user_indexer.labels,
        )
        self.inv_item_indexer = IndexToString(
            inputCol="item_idx",
            outputCol="item_id",
            labels=self.item_indexer.labels,
        )

    @abstractmethod
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Inner method where model actually fits.

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features
            ``[user_id, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id, timestamp]`` + feature columns
        :return:
        """

    # pylint: disable=too-many-arguments
    def _predict_wrap(
        self,
        log: Optional[AnyDataFrame],
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Predict wrapper to allow for fewer parameters in models

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param users: users to create recommendations for
            dataframe containing ``[user_id]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_id]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be``0``.
        :param user_features: user features
            ``[user_id , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_id, item_id, relevance]``
        """
        self.logger.debug("Starting predict %s", type(self).__name__)

        log, user_features, item_features = [
            convert2spark(df) for df in [log, user_features, item_features]
        ]

        user_data = users or log or user_features or self.user_indexer.labels
        users = self._get_ids(user_data, "user_id")

        item_data = items or log or item_features or self.item_indexer.labels
        items = self._get_ids(item_data, "item_id")

        users_type = users.schema["user_id"].dataType
        items_type = items.schema["item_id"].dataType

        log, user_features, item_features, users, items = [
            self._convert_index(df)
            for df in [log, user_features, item_features, users, items]
        ]

        num_items = items.count()
        if num_items < k:
            raise ValueError(
                f"k = {k} > number of items = {num_items}"
            )

        recs = self._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        if filter_seen_items and log:
            recs = recs.join(
                log.withColumnRenamed("item_idx", "item")
                .withColumnRenamed("user_idx", "user")
                .select("user", "item"),
                on=(sf.col("user_idx") == sf.col("user"))
                & (sf.col("item_idx") == sf.col("item")),
                how="anti",
            ).drop("user", "item")

        recs = self._convert_back(recs, users_type, items_type).select(
            "user_id", "item_id", "relevance"
        )
        recs = get_top_k_recs(recs, k)
        return recs

    def _convert_index(
        self, data_frame: Optional[DataFrame]
    ) -> Optional[DataFrame]:
        """
        Convert raw ``user_id`` and ``item_id`` to numerical ``user_idx`` and ``item_idx``

        :param data_frame: dataframe with raw indexes
        :return: dataframe with converted indexes
        """
        if data_frame is None:
            return None
        if "user_id" in data_frame.columns:
            self._reindex("user", data_frame)
            data_frame = self.user_indexer.transform(data_frame).drop(
                "user_id"
            )
            data_frame = data_frame.withColumn(
                "user_idx", sf.col("user_idx").cast("int")
            )
        if "item_id" in data_frame.columns:
            self._reindex("item", data_frame)
            data_frame = self.item_indexer.transform(data_frame).drop(
                "item_id"
            )
            data_frame = data_frame.withColumn(
                "item_idx", sf.col("item_idx").cast("int")
            )
        return data_frame

    def _convert_back(self, log, user_type, item_type):
        res = log
        if "user_idx" in log.columns:
            res = (
                self.inv_user_indexer.transform(res)
                .drop("user_idx")
                .withColumn("user_id", sf.col("user_id").cast(user_type))
            )
        if "item_idx" in log.columns:
            res = (
                self.inv_item_indexer.transform(res)
                .drop("item_idx")
                .withColumn("item_id", sf.col("item_id").cast(item_type))
            )
        return res

    def _reindex(self, entity: str, objects: DataFrame):
        """
           Reindex users or items. If recommender can process cold entities,
           indexer is updated with new entries.

           :param entity: user or item
           :param objects: unique users/items
        """
        indexer = getattr(self, f"{entity}_indexer")
        inv_indexer = getattr(self, f"inv_{entity}_indexer")
        can_reindex = getattr(self, f"can_predict_cold_{entity}s")
        new_objects = set(
            map(
                str,
                objects.select(sf.collect_list(indexer.getInputCol())).first()[
                    0
                ],
            )
        ).difference(indexer.labels)
        if new_objects:
            if can_reindex:
                new_labels = indexer.labels + list(new_objects)
                setattr(
                    self,
                    f"{entity}_indexer",
                    indexer.from_labels(
                        new_labels,
                        inputCol=indexer.getInputCol(),
                        outputCol=indexer.getOutputCol(),
                        handleInvalid="error",
                    ),
                )
                inv_indexer.setLabels(new_labels)
            else:
                message = (
                    f"{entity} contains cold elements, recommendations won't be complete."
                )
                self.logger.warning(message)
                indexer.setHandleInvalid("skip")

    @staticmethod
    def _get_ids(
        log: Union[Iterable, AnyDataFrame], column: str,
    ) -> DataFrame:
        """
        Get unique values from ``array`` and put them into dataframe with column ``column``.
        """
        spark = State().session
        if isinstance(log, DataFrame):
            unique = log.select(column).distinct()
        elif isinstance(log, collections.abc.Iterable):
            unique = spark.createDataFrame(
                data=pd.DataFrame(pd.unique(list(log)), columns=[column])
            )
        else:
            raise ValueError("Wrong type %s" % type(log))
        return unique

    # pylint: disable=too-many-arguments
    @abstractmethod
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
        """
        Inner method where model actually predicts.

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param users: users to create recommendations for
            dataframe containing ``[user_id]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_id]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be``0``.
        :param user_features: user features
            ``[user_id , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_id, item_id, relevance]``
        """

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    @property
    def users_count(self) -> int:
        """
        :returns: number of users the model was trained on
        """
        try:
            return len(self.user_indexer.labels)
        except AttributeError:
            raise AttributeError(
                "Must run fit before calling this method"
            )

    @property
    def items_count(self) -> int:
        """
        :returns: number of items the model was trained on
        """
        try:
            return len(self.item_indexer.labels)
        except AttributeError:
            raise AttributeError(
                "Must run fit before calling this method"
            )

    def _fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
        force_reindex: bool = True,
    ) -> DataFrame:
        self._fit_wrap(log, user_features, item_features, force_reindex)
        return self._predict_wrap(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )

    def _clear_cache(self):
        """
        Clear spark cache
        """

    def _predict_pairs_wrap(
        self,
        pairs: AnyDataFrame,
        log: Optional[AnyDataFrame] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        """
        This method
        1) converts data to spark
        2) converts indexes
        3) calls inner _predict_pairs method of a model
        4) converts indexes back

        :param pairs: user-item pairs to get relevance for,
            dataframe containing``[user_id, item_id]``.
        :param log: train data
            ``[user_id, item_id, timestamp, relevance]``.
        :return: recommendations
            ``[user_id, item_id, relevance]`` for given pairs
        """
        log, user_features, item_features, pairs = [
            convert2spark(df)
            for df in [log, user_features, item_features, pairs]
        ]
        if sorted(pairs.columns) != ["item_id", "user_id"]:
            raise ValueError(
                "pairs must be a dataframe with columns strictly [user_id, item_id]"
            )

        users_type = pairs.schema["user_id"].dataType
        items_type = pairs.schema["item_id"].dataType

        log, user_features, item_features, pairs = [
            self._convert_index(df)
            for df in [log, user_features, item_features, pairs]
        ]

        pred = self._predict_pairs(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

        pred = self._convert_back(pred, users_type, items_type).select(
            "user_id", "item_id", "relevance"
        )
        return pred

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Fallback method to use in case ``_predict_pairs`` is not implemented.
        Simply joins ``predict`` with given ``pairs``.
        :param pairs: user-item pairs to get relevance for,
            dataframe containing``[user_idx, item_idx]``.
        :param log: train data
            ``[user_idx, item_idx, timestamp, relevance]``.
        :return: recommendations
            ``[user_idx, item_idx, relevance]`` for given pairs
        """
        message = (
            "native predict_pairs is not implemented for this model. "
            "Falling back to usual predict method and filtering the results."
        )
        self.logger.warning(message)

        users = pairs.select("user_idx").distinct()
        items = pairs.select("item_idx").distinct()
        k = items.count()
        pred = self._predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=False,
        )

        pred = pred.join(
            pairs.select("user_idx", "item_idx"),
            on=["user_idx", "item_idx"],
            how="inner",
        )
        return pred

    def _get_features_wrap(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Optional[Tuple[DataFrame, int]]:
        if "user_id" not in ids.columns and "item_id" not in ids.columns:
            raise ValueError(
                "user_id or item_id missing"
            )

        idx_col_name = "item_id" if "item_id" in ids.columns else "user_id"

        ids_type = ids.schema[idx_col_name].dataType
        ids, features = [self._convert_index(df) for df in [ids, features]]

        vectors, rank = self._get_features(ids, features)
        vectors = self._convert_back(
            log=vectors,
            user_type=ids_type if idx_col_name == "user_id" else None,
            item_type=ids_type if idx_col_name == "item_id" else None,
        )

        return vectors, rank

    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Optional[Tuple[DataFrame, int]]:
        raise NotImplementedError(
            "Method is implemented only for ALS and LightFMWrap"
        )


# pylint: disable=abstract-method
class HybridRecommender(BaseRecommender, ABC):
    """Рекомендатель, учитывающий фичи"""

    def fit(
        self,
        log: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        force_reindex: bool = True,
    ) -> None:
        """
        Обучает модель на логе и признаках пользователей и объектов.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id, timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id, timestamp]`` и колонки с признаками
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=user_features,
            item_features=item_features,
            force_reindex=force_reindex,
        )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Выдача рекомендаций для пользователей.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]`` или ``array-like``;
            если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_id]`` или ``array-like``;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в ``relevance`` в рекомендациях к ним будет стоять ``0``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )

    def fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
        force_reindex: bool = True,
    ) -> DataFrame:
        """
        Обучает модель и выдает рекомендации.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то поднимается исключение
        :param items: список объектов, которые необходимо рекомендовать;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять ``0``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param filter_seen_items: если ``True``, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
            force_reindex=force_reindex,
        )

    def predict_pairs(
        self,
        pairs: AnyDataFrame,
        log: Optional[AnyDataFrame] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        """
        Возвращает релевантности для конкретных пар user-item, переданных в pairs.
        В случае, если модель не вернула relevance для каких-то из пар,
        они исключаются из возвращаемого датафрейма.

        :param pairs: пары пользователь-объект, для которых необходимо получить relevance,
            spark- или pandas-датафрейм с колонками ``[user_id, item_id]``
        :param log: лог взаимодействий пользователей и объектов,
            spark- или pandas-датафрейм с колонками ``[user_id, item_id, timestamp, relevance]``.
            Необходим для корректной работы inference некоторых алгоритмов
        :param user_features: spark- или pandas-датафрейм, содержащий признаки пользователей и user_id
        :param item_features: spark- или pandas-датафрейм, содержащий признаки объектов и item_id
        :return: рекомендации для переданных пар, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._predict_pairs_wrap(
            pairs, log, user_features, item_features
        )

    def get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Optional[Tuple[DataFrame, int]]:
        """
        Возвращает вектора пользователей или объектов в виде столбца с типом ArrayType
        :param ids: spark-датафрейм с уникальными id пользователей или объектов
        :param features: spark-датафрейм c признаками пользователей или объектов, для которых переданы id
        :return: вектора пользователей или объектов.
            Если модель не может вернуть вектор для какого-то id, id исключается из датафрейма с результатами
        """
        return self._get_features_wrap(ids, features)


# pylint: disable=abstract-method
class Recommender(BaseRecommender, ABC):
    """Обычный рекомендатель"""

    def fit(self, log: AnyDataFrame, force_reindex: bool = True) -> None:
        """
        Обучает модель на логе и признаках пользователей и объектов.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=None,
            item_features=None,
            force_reindex=force_reindex,
        )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Выдача рекомендаций для пользователей.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``.
            Необходим некоторым моделям для предикта и используется для фильтрации просмотренных объектов.
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]`` или ``array-like``;
            если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_id]`` или ``array-like``;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в ``relevance`` в рекомендациях к ним будет стоять ``0``
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
        )

    def predict_pairs(
        self, pairs: AnyDataFrame, log: Optional[AnyDataFrame] = None
    ) -> DataFrame:
        """
        Возвращает релевантности для конкретных пар user-item, переданных в pairs.
        В случае, если модель не вернула relevance для каких-то из пар,
        они исключаются из возвращаемого датафрейма.

        :param pairs: пары пользователь-объект, для которых необходимо получить relevance,
            spark- или pandas-датафрейм с колонками ``[user_id, item_id]``
        :param log: лог взаимодействий пользователей и объектов,
            spark- или pandas-датафрейм с колонками ``[user_id, item_id, timestamp, relevance]``.
            Необходим для корректной работы inference некоторых алгоритмов
        :return: рекомендации для переданных пар, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._predict_pairs_wrap(pairs, log, None, None)

    # pylint: disable=too-many-arguments
    def fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        force_reindex: bool = True,
    ) -> DataFrame:
        """
        Обучает модель и выдает рекомендации.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то поднимается исключение
        :param items: список объектов, которые необходимо рекомендовать;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять ``0``
        :param filter_seen_items: если ``True``, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
            force_reindex=force_reindex,
        )

    def get_features(self, ids: DataFrame) -> Optional[Tuple[DataFrame, int]]:
        """
        Возвращает вектора пользователей или объектов в виде столбца с типом ArrayType

        :param ids: spark-датафрейм с уникальными id пользователей или объектов, колонкой user_id или item_id
        :return: вектора пользователей или объектов.
            Если модель не может вернуть вектор для какого-то id, id исключается из датафрейма с результатами
        """
        return self._get_features_wrap(ids, None)


class UserRecommender(BaseRecommender, ABC):
    """Использует фичи пользователей, но не использует фичи айтемов. Лог — необязательный параметр."""

    def fit(
        self,
        log: AnyDataFrame,
        user_features: AnyDataFrame,
        force_reindex: bool = True,
    ) -> None:
        """
        Выделить кластеры и посчитать популярность объектов в них.

        :param log: логи пользователей с историей для подсчета популярности объектов
        :param user_features: датафрейм связывающий `user_id` пользователей и их числовые признаки
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        """
        self._fit_wrap(
            log=log, user_features=user_features, force_reindex=force_reindex
        )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        user_features: AnyDataFrame,
        k: int,
        log: Optional[AnyDataFrame] = None,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Получить предсказания для переданных пользователей

        :param user_features: айди пользователей с числовыми фичами
        :param k: длина рекомендаций
        :param log: опциональный датафрейм с логами пользователей.
            Если передан, объекты отсюда удаляются из рекомендаций для соответствующих пользователей.
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если ``None``, выбираются все пользователи из лога;
        :param items: список объектов, которые необходимо рекомендовать;
            если ``None``, выбираются все объекты из лога;
        :param filter_seen_items: если ``True``, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        :return: датафрейм с рекомендациями
        """
        return self._predict_wrap(
            log=log,
            user_features=user_features,
            k=k,
            filter_seen_items=filter_seen_items,
            users=users,
            items=items,
        )

    def predict_pairs(
        self,
        pairs: AnyDataFrame,
        log: Optional[AnyDataFrame] = None,
        user_features: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        """
        Возвращает релевантности для конкретных пар user-item, переданных в pairs.
        В случае, если модель не вернула relevance для каких-то из пар,
        они исключаются из возвращаемого датафрейма.

        :param pairs: пары пользователь-объект, для которых необходимо получить relevance,
            spark- или pandas-датафрейм с колонками ``[user_id, item_id]``
        :param log: лог взаимодействий пользователей и объектов,
            spark- или pandas-датафрейм с колонками ``[user_id, item_id, timestamp, relevance]``.
            Необходим для корректной работы inference некоторых алгоритмов
        :param user_features: spark- или pandas-датафрейм, содержащий признаки пользователей и user_id
        :return: рекомендации для переданных пар, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._predict_pairs_wrap(pairs, log, user_features, None)


class NeighbourRec(Recommender, ABC):
    """ Базовый класс для алгоритмов, использующих join матрицы сходства объектов с логом на inference"""

    similarity: Optional[DataFrame]

    def _clear_cache(self):
        if hasattr(self, "similarity"):
            self.similarity.unpersist()

    def _predict_pairs_inner(
        self,
        log: DataFrame,
        filter_df: DataFrame,
        condition: Column,
        users: DataFrame,
    ) -> DataFrame:
        """
        Получение рекомендаций для всех выбранных пользователей
        с фильтрацией объектов путем inner join промежуточных результатов с filter_df по condition.
        Позволяет реализовать как predict для пар, так и обычный predict top-k.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками ``[user_idx, item_idx, timestamp, relevance]``.
        :param filter_df: спарк-датафрейм, по которому будет выполняться фильтрация объектов,
            спарк-датафрейм с колонками ``[item_idx_filter]`` или ``[user_idx_filter, item_idx_filter]``.
        :param condition: условие для inner join датафрейма, полученного перед подсчетом relevance и filter_df
        :param users: пользователи, для которых нужно получить рекомендации
        :return: спарк-датафрейм с колонками ``[user_idx, item_idx, relevance]``
        """
        if log is None:
            raise ValueError(
                "Для predict {} необходим log.".format(self.__str__())
            )

        recs = (
            log.join(users, how="inner", on="user_idx")
            .join(
                self.similarity,
                how="inner",
                on=sf.col("item_idx") == sf.col("item_id_one"),
            )
            .join(filter_df, how="inner", on=condition,)
            .groupby("user_idx", "item_id_two")
            .agg(sf.sum("similarity").alias("relevance"))
            .withColumnRenamed("item_id_two", "item_idx")
        )
        return recs

    # pylint: disable=too-many-arguments
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
        return self._predict_pairs_inner(
            log=log,
            filter_df=items.withColumnRenamed("item_idx", "item_idx_filter"),
            condition=sf.col("item_id_two") == sf.col("item_idx_filter"),
            users=users,
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return self._predict_pairs_inner(
            log=log,
            filter_df=(
                pairs.withColumnRenamed(
                    "user_idx", "user_idx_filter"
                ).withColumnRenamed("item_idx", "item_idx_filter")
            ),
            condition=(sf.col("user_idx") == sf.col("user_idx_filter"))
            & (sf.col("item_id_two") == sf.col("item_idx_filter")),
            users=pairs.select("user_idx").distinct(),
        )
