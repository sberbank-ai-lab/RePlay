from typing import Dict, Optional, Tuple, List, Union, Any

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, Window

from replay.constants import AnyDataFrame
from replay.metrics import Metric, Precision
from replay.models import ALSWrap, PopRec, LightFMWrap
from replay.models.base_rec import BaseRecommender

from replay.utils import (
    array_mult,
    fallback,
    get_top_k_recs,
    horizontal_explode,
    join_or_return,
    ugly_join,
)


def add_labels(
    prediction: DataFrame, ground_truth: DataFrame, label_name: str = "target"
) -> DataFrame:
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

    return prediction.join(
        ground_truth.select("user_idx", "item_idx").withColumn(
            label_name, sf.lit(1.0)
        ),
        on=["user_idx", "item_idx"],
        how="left",
    ).fillna(0.0, subset=label_name)


def remove_features_for_lightfm(
    model: BaseRecommender, features: Optional[DataFrame] = None
) -> Optional[DataFrame]:
    """
    Use LightFM without features in ensemble
    :param model: RePlay model
    :param features: dataframe with features
    :return: None if model is LightFM and features otherwise
    """
    if not isinstance(model, LightFMWrap):
        return features
    return None


# pylint: disable=too-many-locals, too-many-arguments
def get_model_vectors(
    model: BaseRecommender,
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
        users, remove_features_for_lightfm(model, user_features)
    )
    item_factors, item_vector_len = model._get_features_wrap(
        items, remove_features_for_lightfm(model, item_features)
    )

    pairs_with_features = join_or_return(
        pairs, user_factors, how="left", on="user_idx"
    )
    pairs_with_features = join_or_return(
        pairs_with_features,
        item_factors,
        how="left",
        on="item_idx",
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
class RecEnsemble:
    """
    The class is used to combine recommendations from different sources (models) and generate
    model-related features, e.g relevance, ranks, vectors.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        models: Union[List[BaseRecommender], BaseRecommender] = ALSWrap(
            rank=128
        ),
        fallback_model: Optional[BaseRecommender] = PopRec(),
    ) -> None:
        """
        :param models: model or a list of models to use
        :param fallback_model: model used to add recommendations
            for users with insufficient recommendations number
            (use it to get exactly num_recs_by_one_model * len(models) recommendations
            for each user)
        """
        self.models = (
            [models] if isinstance(models, BaseRecommender) else models
        )
        self.fallback_model = fallback_model

    @staticmethod
    def _predict_with_first_level_model(
        model: BaseRecommender,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
        log_to_filter: DataFrame,
        filter_seen_items: bool,
    ):
        """
        The method is used to predict with one log and filter seen from another log.
        """
        max_positives_to_filter = 0

        log_to_filter_cached = ugly_join(
            left=log_to_filter,
            right=users,
            on_col_name="user_idx",
        ).cache()

        if filter_seen_items:
            if log_to_filter_cached.count() > 0:
                max_positives_to_filter = (
                    log_to_filter_cached.groupBy("user_idx")
                    .agg(sf.count("item_idx").alias("num_positives"))
                    .select(sf.max("num_positives"))
                    .collect()[0][0]
                )

        pred = model._predict_wrap(
            log=log,
            k=k + max_positives_to_filter,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=False,
        )

        if filter_seen_items:
            pred = pred.join(
                log_to_filter_cached.select("user_idx", "item_idx"),
                on=["user_idx", "item_idx"],
                how="anti",
            )

        log_to_filter_cached.unpersist()
        return get_top_k_recs(pred, k)

    # pylint: disable=too-many-locals,too-many-statements
    def fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Fit `models` and `fallback_model`

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx, timestamp]`` + feature columns
        :return:
        """
        for base_model in [
            *self.models,
            self.fallback_model,
        ]:
            base_model._fit_wrap(
                log=log,
                user_features=remove_features_for_lightfm(
                    model=base_model, features=user_features
                ),
                item_features=remove_features_for_lightfm(
                    model=base_model, features=item_features
                ),
            )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: DataFrame,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        num_recs_by_one_model: Union[int, List[int]] = 100,
        filter_seen_items: bool = True,
        log_to_filter: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Get recommendations.
        Number of recommendations per one model is defined by num_recs_by_one_model.
        Duplicated recommendations removed. If fallback model is defined, it is used to
        produce additional recommendations and return exactly sum(num_recs_by_one_model)
        recommendations for each user.

        :param log: historical log of interactions used for prediction
            ``[user_idx, item_idx, timestamp, relevance]``
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param num_recs_by_one_model: number of recommendations produced by each model
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param log_to_filter: interactions to filter from the recommendations
        :return: recommendations dataframe
            ``[user_idx, item_idx, <columns with relevance for each model>]``
        """

        # TO DO remove log_to_filter and _predict_with_first_level_model if not required
        if log_to_filter is None:
            log_to_filter = log

        num_recs_by_one_model = (
            [num_recs_by_one_model] * len(self.models)
            if isinstance(num_recs_by_one_model, int)
            else num_recs_by_one_model
        )

        full_pred = None
        for idx, model in enumerate(self.models):
            if num_recs_by_one_model[idx] > 0:
                current_pred = self._predict_with_first_level_model(
                    model=model,
                    log=log,
                    k=num_recs_by_one_model[idx],
                    users=users,
                    items=items,
                    user_features=remove_features_for_lightfm(
                        model, user_features
                    ),
                    item_features=remove_features_for_lightfm(
                        model, item_features
                    ),
                    log_to_filter=log_to_filter,
                    filter_seen_items=filter_seen_items,
                ).withColumnRenamed("relevance", f"rel_{idx}_{model}")

                if full_pred is None:
                    full_pred = current_pred
                else:
                    full_pred = full_pred.join(
                        current_pred, on=["user_idx", "item_idx"], how="outer"
                    )

        if self.fallback_model is None:
            return full_pred

        fallback_candidates = self._predict_with_first_level_model(
            model=self.fallback_model,
            log=log,
            k=sum(num_recs_by_one_model),
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            log_to_filter=log_to_filter,
            filter_seen_items=filter_seen_items,
        )

        fb_pred = fallback(
            base=full_pred.select(
                "user_idx", "item_idx", sf.lit(1).alias("relevance")
            ),
            fill=fallback_candidates.withColumn("rel_fb", sf.col("relevance")),
            k=sum(num_recs_by_one_model),
        ).drop("relevance")

        return fb_pred.join(full_pred, on=["user_idx", "item_idx"], how="left")

    def add_relevance_vectors(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame],
        user_features: Optional[DataFrame],
        item_features: Optional[DataFrame],
        calc_relevance: Union[bool, List[bool]] = True,
        calc_rank: bool = True,
        add_features: Union[bool, List[bool]] = False,
    ):
        """
        Calculate relevance and rank for user-item pairs,
        add user/item vectors created by the models.

        :param pairs: user-item pairs to add relevance/rank/vectors
        :param log: historical log of interactions used for relevance calculation
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param calc_relevance: flag of list of flags indicating if add relevance,
            calculated by all models/each model
        :param calc_rank: flag of list of flags indicating if add rank,
            calculated by all models/each model.
            Rank is a position of item in user's recommendations sorted by
            relevance in descending order
        :param add_features: flag of list of flags indicating if add vectors,
            produced by all models/each model.
            Some models, e.g. ALS, LightFM, Word2Vec, generate user or item vectors
            which could be used by a re-ranking model.
        :return: pred dataframe with generated feature columns
        """
        if isinstance(calc_relevance, bool):
            calc_relevance = [calc_relevance] * len(self.models)

        if any(calc_relevance):
            for idx, flag in enumerate(calc_relevance):
                if flag:
                    to_add = self.models[idx]._predict_pairs_wrap(
                        pairs.select("user_idx", "item_idx"),
                        log,
                        remove_features_for_lightfm(
                            self.models[idx], user_features
                        ),
                        remove_features_for_lightfm(
                            self.models[idx], item_features
                        ),
                    )
                    pairs = pairs.join(
                        to_add.withColumnRenamed(
                            "relevance", f"rel_{idx}_{self.models[idx]}_all"
                        ),
                        on=["user_idx", "item_idx"],
                        how="left",
                    )
                    if calc_rank:
                        pairs = pairs.withColumn(
                            f"rank_{idx}_{self.models[idx]}_all",
                            sf.row_number().over(
                                Window.partitionBy("user_idx").orderBy(
                                    sf.col(
                                        f"rel_{idx}_{self.models[idx]}_all"
                                    ).desc()
                                )
                            ),
                        )
        pairs_cached = pairs.cache()
        if isinstance(add_features, bool):
            add_features = [add_features] * len(self.models)

        if any(add_features):
            for idx, flag in enumerate(add_features):
                if flag:
                    pairs_cached = get_model_vectors(
                        model=self.models[idx],
                        pairs=pairs_cached,
                        user_features=remove_features_for_lightfm(
                            self.models[idx], user_features
                        ),
                        item_features=remove_features_for_lightfm(
                            self.models[idx], item_features
                        ),
                        add_factors_mult=True,
                        prefix=f"m_{idx}",
                    )

        pairs_cached.unpersist()
        return pairs_cached

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
        param_borders: Optional[List[Optional[Dict[str, List[Any]]]]] = None,
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
        :param param_borders: None or a list with param grids for the models and fallback model.
            None means optimize each model with default search space.
            The elements of list could be:
            - a dict of``{param: [low, high]}``, which will be used for optimization
            - empty dict, which means skipping optimization for that model
            - None, which means using default search space for that model
        :param criterion: metric to optimize
        :param k: length of a recommendation list
        :param budget: number of optuna trials to train each model
        :param new_study: keep searching with previous study or start a new study
        :return: list of dicts of parameters
        """
        number_of_models = len(self.models)
        if self.fallback_model is not None:
            number_of_models += 1
        if param_borders is None:
            param_borders = [None] * number_of_models
        if number_of_models != len(param_borders):
            raise ValueError(
                "Provide search grid or None for each of first level models and fallback model"
            )

        params_found = []
        for i, model in enumerate(self.models):
            if param_borders[i] is None or (
                isinstance(param_borders[i], dict) and param_borders[i]
            ):
                params_found.append(
                    model.optimize(
                        train=train,
                        test=test,
                        user_features=remove_features_for_lightfm(
                            model, user_features
                        ),
                        item_features=remove_features_for_lightfm(
                            model, item_features
                        ),
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

        fallback_params = self._optimize_one_model(
            model=self.fallback_model,
            train=train,
            test=test,
            user_features=remove_features_for_lightfm(
                self.fallback_model, user_features
            ),
            item_features=remove_features_for_lightfm(
                self.fallback_model, item_features
            ),
            param_borders=param_borders[-1],
            criterion=criterion,
            new_study=new_study,
        )
        return params_found, fallback_params
