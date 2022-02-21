# pylint: disable=too-many-lines
from collections.abc import Iterable
from typing import Dict, Optional, Tuple, List, Union, Any

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from replay.constants import AnyDataFrame
from replay.data_preparator import ToNumericFeatureTransformer
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.metrics import Metric, Precision
from replay.models import ALSWrap, RandomRec, PopRec
from replay.models.base_rec import BaseRecommender, HybridRecommender
from replay.scenarios.two_stages.reranker import LamaWrap

from replay.session_handler import State
from replay.splitters import Splitter, UserSplitter
from replay.utils import (
    array_mult,
    cache_if_exists,
    fallback,
    get_log_info,
    get_top_k_recs,
    horizontal_explode,
    join_or_return,
    ugly_join,
    unpersist_if_exists,
)


def add_labels(prediction: DataFrame, ground_truth: DataFrame, label_name: str = "target") -> DataFrame:
    """
    Add a column with class labels named as the `target_name`.
    User-item pairs from `prediction` get label `1` if present in `ground_truth` otherwise `0`.
    >>> import pandas as pd
    >>> prediction = pd.DataFrame({"user_idx": [1, 1, 2], "item_idx": [1, 2, 2]})
    >>> prediction
       user_idx  item_idx
    0         1         1
    1         1         2
    2         2         2

    >>> ground_truth = pd.DataFrame({"user_idx": [1, 1, 3], "item_idx": [1, 4, 5]})
    >>> ground_truth
       user_idx  item_idx
    0         1         1
    1         1         4
    2         3         5

    >>> from replay.utils import convert2spark
    >>> prediction = convert2spark(prediction)
    >>> ground_truth = convert2spark(ground_truth)

    >>> res = add_labels(prediction, ground_truth, label_name='class_label')
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  class_label
    0         1         1          1.0
    1         1         2          0.0
    2         2         2          0.0

    :param prediction: spark dataframe with columns `[user-idx, item_idx, <feature_columns, ...>]`
        to add class labels
    :param ground_truth: spark dataframe with columns `[user-idx, item_idx]`.
        Contains ground truth, actual user-item interactions.
    :param label_name: name of a column with class labels which will be added
    :return: labeled `prediction`
    """

    return (
        prediction.join(
            ground_truth.select(
                "user_idx", "item_idx"
            ).withColumn(label_name, sf.lit(1.0)),
            on=["user_idx", "item_idx"],
            how="left",
        ).fillna(0.0, subset=label_name)
    )


# pylint: disable=too-many-locals, too-many-arguments
def get_first_level_model_features(
    model: DataFrame,
    pairs: DataFrame,
    user_features: Optional[DataFrame] = None,
    item_features: Optional[DataFrame] = None,
    add_factors_mult: bool = True,
    prefix: str = "",
) -> DataFrame:
    """
    Get user and item embeddings from replay model.
    Can also compute elementwise multiplication between them with ``add_factors_mult`` parameter.
    Zero vectors are returned if a model does not have embeddings for specific users/items.

    :param model: trained model
    :param pairs: user-item pairs to get vectors for `[user_idx, item_idx]`
    :param user_features: user features `[user_idx, feature_1, ....]`
    :param item_features: item features `[item_idx, feature_1, ....]`
    :param add_factors_mult: flag to add elementwise multiplication
    :param prefix: name to add to the columns
    :return: DataFrame
    """
    users = pairs.select("user_idx").distinct()
    items = pairs.select("item_idx").distinct()
    user_factors, user_vector_len = model._get_features_wrap(
        users, user_features
    )
    item_factors, item_vector_len = model._get_features_wrap(
        items, item_features
    )

    pairs_with_features = join_or_return(
        pairs, user_factors, how="left", on="user_idx"
    )
    pairs_with_features = join_or_return(
        pairs_with_features, item_factors, how="left", on="item_idx",
    )

    factors_to_explode = []
    if user_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "user_factors",
            sf.coalesce(
                sf.col("user_factors"),
                sf.array([sf.lit(0.0)] * user_vector_len),
            ),
        )
        factors_to_explode.append(("user_factors", "uf"))

    if item_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "item_factors",
            sf.coalesce(
                sf.col("item_factors"),
                sf.array([sf.lit(0.0)] * item_vector_len),
            ),
        )
        factors_to_explode.append(("item_factors", "if"))

    if model.__str__() == "LightFMWrap":
        pairs_with_features = (
            pairs_with_features.fillna({"user_bias": 0, "item_bias": 0})
            .withColumnRenamed("user_bias", f"{prefix}_user_bias")
            .withColumnRenamed("item_bias", f"{prefix}_item_bias")
        )

    if (
        add_factors_mult
        and user_factors is not None
        and item_factors is not None
    ):
        pairs_with_features = pairs_with_features.withColumn(
            "factors_mult",
            array_mult(sf.col("item_factors"), sf.col("user_factors")),
        )
        factors_to_explode.append(("factors_mult", "fm"))

    for col_name, feature_prefix in factors_to_explode:
        col_set = set(pairs_with_features.columns)
        col_set.remove(col_name)
        pairs_with_features = horizontal_explode(
            data_frame=pairs_with_features,
            column_to_explode=col_name,
            other_columns=[sf.col(column) for column in sorted(list(col_set))],
            prefix=f"{prefix}_{feature_prefix}",
        )

    return pairs_with_features


# pylint: disable=too-many-instance-attributes
class RecAnsamble(HybridRecommender):
    """
    1) train ``models``
    2) create negative examples to train second stage model using one of:

       - wrong recommendations from first stage
       - random examples

        use ``num_recs_by_one_model`` to specify number of negatives per user
    4) augments dataset with features:

       - get 1 level recommendations for positive examples
         from second_level_train and for generated negative examples
       - add user and item features
       - generate statistical and pair features

    5) train ``TabularAutoML`` from LightAutoML

    *inference*:

    1) take ``log``
    2) generate candidates, their number can be specified with ``num_recs_by_one_model``
    3) add features as in train
    4) get recommendations

    """

    can_predict_cold_users: bool = True
    can_predict_cold_items: bool = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        models: Union[
            List[BaseRecommender], BaseRecommender
        ] = ALSWrap(rank=128),
        fallback_model: Optional[BaseRecommender] = PopRec(),
        num_recs_by_one_model: Union[int, List[int]] = 100,
        add_models_feat: Union[bool, List[bool]] = False,
        use_user_item_features: Union[bool, List[bool]] = False,
        seed: int = 123
    ) -> None:
        """
        :param models: model or a list of models to use
        :param fallback_model: model used to add recommendations
            for users with insufficient recommendations number
            (use it to get exactly num_recs_by_one_model * len(models) recommendations for each user)
        :param num_recs_by_one_model: number of recommendations generated by each model.
            Fallback model generates num_recs_by_one_model * len(models) or sum(num_recs_by_one_model)
            recommendations for each user
        :param add_models_feat: flag or a list of flags to use
            features created by first level models
        :param seed: random seed

        """
        self.first_level_models = (
            models
            if isinstance(models, Iterable)
            else [models]
        )
        self.fallback_model = fallback_model

        if isinstance(add_models_feat, bool):
            self.add_models_feat = [
                add_models_feat
            ] * len(self.first_level_models)
        else:
            if len(self.first_level_models) != len(
                add_models_feat
            ):
                raise ValueError(
                    f"For each model from models specify "
                    f"flag to use first level features."
                    f"Length of models is {len(models)}, "
                    f"Length of add_models_feat is {len(add_models_feat)}"
                )

            self.add_models_feat = add_models_feat

        self.use_features = (use_user_item_features
                             if isinstance(use_user_item_features, Iterable)
                             else [use_user_item_features]
                )
        self.num_recs_by_one_model = num_recs_by_one_model
        self.add_models_feat = add_models_feat
        self.seed = seed

    # pylint: disable=too-many-locals
    def _add_features_for_second_level(
        self,
        log_to_add_features: DataFrame,
        log_for_first_level_models: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ) -> DataFrame:
        """
        Added features are:
            - relevance from first level models
            - user and item features from first level models
            - dataset features
            - FeatureProcessor features

        :param log_to_add_features: input DataFrame``[user_idx, item_idx, timestamp, relevance]``
        :param log_for_first_level_models: DataFrame``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features``[user_idx]`` + feature columns
        :param item_features: item features``[item_idx]`` + feature columns
        :return: DataFrame
        """
        self.logger.info("Generating features")
        full_second_level_train = log_to_add_features
        first_level_item_features_cached = cache_if_exists(
            self.first_level_item_features_transformer.transform(item_features)
        )
        first_level_user_features_cached = cache_if_exists(
            self.first_level_user_features_transformer.transform(user_features)
        )

        pairs = log_to_add_features.select("user_idx", "item_idx")
        for idx, model in enumerate(self.first_level_models):
            current_pred = self._predict_pairs_with_first_level_model(
                model=model,
                log=log_for_first_level_models,
                pairs=pairs,
                user_features=first_level_user_features_cached,
                item_features=first_level_item_features_cached,
            ).withColumnRenamed("relevance", f"rel_{idx}_{model}")
            full_second_level_train = full_second_level_train.join(
                sf.broadcast(current_pred),
                on=["user_idx", "item_idx"],
                how="left",
            )

            if self.add_models_feat[idx]:
                features = get_first_level_model_features(
                    model=model,
                    pairs=full_second_level_train.select(
                        "user_idx", "item_idx"
                    ),
                    user_features=first_level_user_features_cached,
                    item_features=first_level_item_features_cached,
                    prefix=f"m_{idx}",
                )
                full_second_level_train = ugly_join(
                    left=full_second_level_train,
                    right=features,
                    on_col_name=["user_idx", "item_idx"],
                    how="left",
                )

        unpersist_if_exists(first_level_user_features_cached)
        unpersist_if_exists(first_level_item_features_cached)

        full_second_level_train_cached = full_second_level_train.fillna(
            0
        ).cache()

        self.logger.info("Adding features from the dataset")
        full_second_level_train = join_or_return(
            full_second_level_train_cached,
            user_features,
            on="user_idx",
            how="left",
        )
        full_second_level_train = join_or_return(
            full_second_level_train, item_features, on="item_idx", how="left",
        )

        if self.add_models_feat:
            if not self.features_processor.fitted:
                self.features_processor.fit(
                    log=log_for_first_level_models,
                    user_features=user_features,
                    item_features=item_features,
                )
            self.logger.info("Adding generated features")
            full_second_level_train = self.features_processor.transform(
                log=full_second_level_train
            )

        self.logger.info(
            "Columns at second level: %s",
            " ".join(full_second_level_train.columns),
        )
        full_second_level_train_cached.unpersist()
        return full_second_level_train

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Write statistics"""
        first_level_train, second_level_train = self.train_splitter.split(log)
        State().logger.debug("Log info: %s", get_log_info(log))
        State().logger.debug(
            "first_level_train info: %s", get_log_info(first_level_train)
        )
        State().logger.debug(
            "second_level_train info: %s", get_log_info(second_level_train)
        )
        return first_level_train, second_level_train

    @staticmethod
    def _filter_or_return(dataframe, condition):
        if dataframe is None:
            return dataframe
        return dataframe.filter(condition)

    def _predict_with_first_level_model(
        self,
        model: BaseRecommender,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
        log_to_filter: DataFrame,
    ):
        """
        Filter users and items using can_predict_cold_items and can_predict_cold_users, and predict
        """
        if not model.can_predict_cold_items:
            log, items, item_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("item_idx") < self.first_level_item_len,
                )
                for df in [log, items, item_features]
            ]
        if not model.can_predict_cold_users:
            log, users, user_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("user_idx") < self.first_level_user_len,
                )
                for df in [log, users, user_features]
            ]

        log_to_filter_cached = ugly_join(
            left=log_to_filter, right=users, on_col_name="user_idx",
        ).cache()
        max_positives_to_filter = 0

        if log_to_filter_cached.count() > 0:
            max_positives_to_filter = (
                log_to_filter_cached.groupBy("user_idx")
                .agg(sf.count("item_idx").alias("num_positives"))
                .select(sf.max("num_positives"))
                .collect()[0][0]
            )

        pred = model._predict(
            log,
            k=k + max_positives_to_filter,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=False,
        )

        pred = pred.join(
            log_to_filter_cached.select("user_idx", "item_idx"),
            on=["user_idx", "item_idx"],
            how="anti",
        ).drop("user", "item")

        log_to_filter_cached.unpersist()

        return get_top_k_recs(pred, k)

    def _predict_pairs_with_first_level_model(
        self,
        model: BaseRecommender,
        log: DataFrame,
        pairs: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ):
        """
        Get relevance for selected user-item pairs.
        """
        if not model.can_predict_cold_items:
            log, pairs, item_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("item_idx") < self.first_level_item_len,
                )
                for df in [log, pairs, item_features]
            ]
        if not model.can_predict_cold_users:
            log, pairs, user_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("user_idx") < self.first_level_user_len,
                )
                for df in [log, pairs, user_features]
            ]

        return model._predict_pairs(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

    # pylint: disable=unused-argument
    def _get_first_level_candidates(
        self,
        model: BaseRecommender,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
        log_to_filter: DataFrame,
    ) -> DataFrame:
        """
        Combining the base model predictions with the fallback model
        predictions.
        """
        passed_arguments = locals()
        passed_arguments.pop("self")
        candidates = self._predict_with_first_level_model(**passed_arguments)

        if self.fallback_model is not None:
            passed_arguments.pop("model")
            fallback_candidates = self._predict_with_first_level_model(
                model=self.fallback_model, **passed_arguments
            )

            candidates = fallback(
                base=candidates,
                fill=fallback_candidates,
                k=self.num_recs_by_one_model,
            )
        return candidates

    # pylint: disable=too-many-locals,too-many-statements
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        item_features_tr, user_features_tr = None, None
        if any(self.use_features):
            self.first_level_item_features_transformer.fit(item_features)
            self.first_level_user_features_transformer.fit(user_features)
            item_features_tr = cache_if_exists(
                self.first_level_item_features_transformer.transform(item_features)
            )
            user_features_tr = cache_if_exists(
                self.first_level_user_features_transformer.transform(user_features)
            )

        for base_model in [
            *self.first_level_models,
            self.fallback_model,
        ]:
            base_model._fit_wrap(
                log=log,
                user_features=user_features_tr,
                item_features=item_features_tr)

        self.logger.info("Generate negative examples")
        negatives_source = (
            self.first_level_models[0]
            if self.negatives_type == "first_level"
            else self.random_model
        )

        first_level_candidates = self._get_first_level_candidates(
            model=negatives_source,
            log=first_level_train,
            k=self.num_recs_by_one_model,
            users=log.select("user_idx").distinct(),
            items=log.select("item_idx").distinct(),
            user_features=user_features_tr,
            item_features=item_features_tr,
            log_to_filter=first_level_train,
        ).select("user_idx", "item_idx")

        unpersist_if_exists(user_features_tr)
        unpersist_if_exists(item_features_tr)

        self.logger.info("Crate train dataset for second level")

        second_level_train = (
            first_level_candidates.join(
                second_level_positive.select(
                    "user_idx", "item_idx"
                ).withColumn("target", sf.lit(1.0)),
                on=["user_idx", "item_idx"],
                how="left",
            ).fillna(0.0, subset="target")
        ).cache()

        self.cached_list.append(second_level_train)

        self.logger.info(
            "Distribution of classes in second-level train dataset:/n %s",
            (
                second_level_train.groupBy("target")
                .agg(sf.count(sf.col("target")).alias("count_for_class"))
                .take(2)
            ),
        )

        self.features_processor.fit(
            log=first_level_train,
            user_features=user_features,
            item_features=item_features,
        )

        self.logger.info("Adding features to second-level train dataset")
        second_level_train_to_convert = self._add_features_for_second_level(
            log_to_add_features=second_level_train,
            log_for_first_level_models=first_level_train,
            user_features=user_features,
            item_features=item_features,
        ).cache()

        self.cached_list.append(second_level_train_to_convert)
        self.second_stage_model.fit(second_level_train_to_convert)
        for dataframe in self.cached_list:
            unpersist_if_exists(dataframe)

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

        State().logger.debug(msg="Generating candidates to rerank")

        first_level_user_features = cache_if_exists(
            self.first_level_user_features_transformer.transform(user_features)
        )
        first_level_item_features = cache_if_exists(
            self.first_level_item_features_transformer.transform(item_features)
        )

        candidates = self._get_first_level_candidates(
            model=self.first_level_models[0],
            log=log,
            k=self.num_recs_by_one_model,
            users=users,
            items=items,
            user_features=first_level_user_features,
            item_features=first_level_item_features,
            log_to_filter=log,
        ).select("user_idx", "item_idx")

        candidates_cached = candidates.cache()
        unpersist_if_exists(first_level_user_features)
        unpersist_if_exists(first_level_item_features)
        self.logger.info("Adding features")
        candidates_features = self._add_features_for_second_level(
            log_to_add_features=candidates_cached,
            log_for_first_level_models=log,
            user_features=user_features,
            item_features=item_features,
        )
        candidates_features.cache()
        candidates_cached.unpersist()
        self.logger.info(
            "Generated %s candidates for %s users",
            candidates_features.count(),
            candidates_features.select("user_idx").distinct().count(),
        )
        return self.second_stage_model.predict(data=candidates_features, k=k)

    def fit_predict(
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
        :param log: input DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param k: length of a recommendation list, must be smaller than the number of ``items``
        :param users: users to get recommendations for
        :param items: items to get recommendations for
        :param user_features: user features``[user_id]`` + feature columns
        :param item_features: item features``[item_id]`` + feature columns
        :param filter_seen_items: flag to removed seen items from recommendations
        :return: DataFrame ``[user_id, item_id, relevance]``
        """
        self.fit(log, user_features, item_features)
        return self.predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )

    @staticmethod
    def _optimize_one_model(
        model: BaseRecommender,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ):
        params = model.optimize(
            train,
            test,
            user_features,
            item_features,
            param_borders,
            criterion,
            k,
            budget,
            new_study,
        )
        return params

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[List[Dict[str, List[Any]]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Optimize first level models with optuna.

        :param train: train DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param test: test DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features ``[user_id , timestamp]`` + feature columns
        :param item_features: item features``[item_id]`` + feature columns
        :param param_borders: list with param grids for first level models and a fallback model.
            Empty dict skips optimization for that model.
            Param grid is a dict ``{param: [low, high]}``.
        :param criterion: metric to optimize
        :param k: length of a recommendation list
        :param budget: number of optuna trials to train each model
        :param new_study: keep searching with previous study or start a new study
        :return: list of dicts of parameters
        """
        number_of_models = len(self.first_level_models)
        if self.fallback_model is not None:
            number_of_models += 1
        if number_of_models != len(param_borders):
            raise ValueError(
                "Provide search grid or None for every first level model"
            )

        params_found = []
        for i, model in enumerate(self.first_level_models):
            if param_borders[i] is None or (
                isinstance(param_borders[i], dict) and param_borders[i]
            ):
                self.logger.info(
                    "Optimizing first level model number %s, %s",
                    i,
                    model.__str__(),
                )
                params_found.append(
                    self._optimize_one_model(
                        model=model,
                        train=train,
                        test=test,
                        user_features=user_features,
                        item_features=item_features,
                        param_borders=param_borders[i],
                        criterion=criterion,
                        k=k,
                        budget=budget,
                        new_study=new_study,
                    )
                )
            else:
                params_found.append(None)

        if self.fallback_model is None or (
            isinstance(param_borders[-1], dict) and not param_borders[-1]
        ):
            return params_found, None

        self.logger.info("Optimizing fallback-model")
        fallback_params = self._optimize_one_model(
            model=self.fallback_model,
            train=train,
            test=test,
            user_features=user_features,
            item_features=item_features,
            param_borders=param_borders[-1],
            criterion=criterion,
            new_study=new_study,
        )
        return params_found, fallback_params
