from typing import Dict, Optional, Tuple, List

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from replay.utils import join_or_return


# pylint: disable=too-many-instance-attributes, too-many-arguments
class TwoStagesFeaturesProcessor:
    """
    Calculate extra features for two stages scenario
    """

    user_log_features: Optional[DataFrame] = None
    item_log_features: Optional[DataFrame] = None
    user_cond_dist_cat_features: Optional[Dict[str, DataFrame]] = None
    item_cond_dist_cat_features: Optional[Dict[str, DataFrame]] = None
    fitted = False

    def __init__(
        self,
        use_log_features: bool = True,
        use_conditional_popularity: bool = True,
        use_cooccurrence: bool = False,
        user_id: str = "user_idx",
        item_id: str = "item_idx",
    ):
        self.use_log_features = use_log_features
        self.use_conditional_popularity = use_conditional_popularity
        self.use_cooccurrence = use_cooccurrence
        self.user_id = user_id
        self.item_id = item_id

    @staticmethod
    def _create_cols_list(log: DataFrame, agg_col: str = "user_idx") -> List:
        """
        Create features based on relevance type
        (binary or not) and whether timestamp is present.
        :param log: input DataFrame ``[user_id(x), item_id(x), timestamp, relevance]``
        :param agg_col: column to create features for, user_id(x) or item_id(x)
        :return: list of columns to pass into pyspark agg
        """
        prefix = agg_col[:1]

        aggregates = [
            sf.log(sf.count(sf.col("relevance"))).alias(
                "{}_log_ratings_count".format(prefix)
            )
        ]

        if (
            log.select(sf.countDistinct(sf.col("timestamp"))).collect()[0][0]
            > 1
        ):
            aggregates.extend(
                [
                    sf.log(sf.countDistinct(sf.col("timestamp"))).alias(
                        "{}_log_rating_dates_count".format(prefix)
                    ),
                    sf.min(sf.col("timestamp")).alias(
                        "{}_min_rating_date".format(prefix)
                    ),
                    sf.max(sf.col("timestamp")).alias(
                        "{}_max_rating_date".format(prefix)
                    ),
                ]
            )

        if (
            log.select(sf.countDistinct(sf.col("relevance"))).collect()[0][0]
            > 1
        ):
            aggregates.extend(
                [
                    (
                        sf.when(
                            sf.stddev(sf.col("relevance")).isNull()
                            | sf.isnan(sf.stddev(sf.col("relevance"))),
                            0,
                        )
                        .otherwise(sf.stddev(sf.col("relevance")))
                        .alias("{}_std".format(prefix))
                    ),
                    sf.mean(sf.col("relevance")).alias(
                        "{}_mean".format(prefix)
                    ),
                ]
            )
            for percentile in [0.05, 0.5, 0.95]:
                aggregates.append(
                    sf.expr(
                        "percentile_approx({}, {})".format(
                            "relevance", percentile
                        )
                    ).alias(
                        "{}_quantile_{}".format(prefix, str(percentile)[2:])
                    )
                )

        return aggregates

    def _calc_log_features(
        self, log: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        user_aggs = self._create_cols_list(log, agg_col=self.user_id)
        user_log_features = log.groupBy(self.user_id).agg(*user_aggs)

        item_aggs = self._create_cols_list(log, agg_col=self.item_id)
        item_log_features = log.groupBy(self.item_id).agg(*item_aggs)

        mean_log_rating_of_user_items = log.join(
            item_log_features.select(self.item_id, "i_log_ratings_count"),
            on=self.item_id,
            how="left",
        )
        mean_log_rating_of_user_items = mean_log_rating_of_user_items.groupBy(
            self.user_id
        ).agg(
            sf.mean("i_log_ratings_count").alias(
                "u_mean_log_items_ratings_count"
            )
        )
        user_log_features = user_log_features.join(
            mean_log_rating_of_user_items, on=self.user_id, how="left"
        )

        mean_log_rating_of_item_users = log.join(
            user_log_features.select(self.user_id, "u_log_ratings_count"),
            on=self.user_id,
            how="left",
        )
        mean_log_rating_of_item_users = mean_log_rating_of_item_users.groupBy(
            self.item_id
        ).agg(
            sf.mean("u_log_ratings_count").alias(
                "i_mean_log_users_ratings_count"
            )
        )
        item_log_features = item_log_features.join(
            mean_log_rating_of_item_users, on=self.item_id, how="left"
        ).cache()

        if "i_mean" in item_log_features.columns:
            # Abnormality: https://hal.inria.fr/hal-01254172/document
            abnormality_df = log.join(
                item_log_features.select(self.item_id, "i_mean", "i_std"),
                on=self.item_id,
                how="left",
            )
            abnormality_df = abnormality_df.withColumn(
                "abnormality", sf.abs(sf.col("relevance") - sf.col("i_mean"))
            )

            abnormality_aggs = [
                sf.mean(sf.col("abnormality")).alias("abnormality")
            ]

            # Abnormality CR: https://hal.inria.fr/hal-01254172/document
            max_std = item_log_features.select(sf.max("i_std")).collect()[0][0]
            min_std = item_log_features.select(sf.min("i_std")).collect()[0][0]
            print(max_std)
            print(min_std)
            if max_std - min_std != 0:
                abnormality_df = abnormality_df.withColumn(
                    "controversy",
                    1
                    - (sf.col("i_std") - sf.lit(min_std))
                    / (sf.lit(max_std - min_std)),
                )
                abnormality_df = abnormality_df.withColumn(
                    "abnormalityCR",
                    (sf.col("abnormality") * sf.col("controversy")) ** 2,
                )
                abnormality_aggs.append(
                    sf.mean(sf.col("abnormalityCR")).alias("abnormalityCR")
                )

            abnormality_res = abnormality_df.groupBy(self.user_id).agg(
                *abnormality_aggs
            )
            user_log_features = user_log_features.join(
                abnormality_res, on=self.user_id, how="left"
            ).cache()

        return user_log_features, item_log_features

    def _add_cond_distr_feat(
        self, cat_cols: List[str], log: DataFrame, features_df: DataFrame
    ) -> Dict[str, DataFrame]:
        """
        Calculate item popularity based on user or item categorical features.
        For example movie popularity among users of the same age.
        If user features are provided, result will contain item features and vice versa.

        :param cat_cols: list of categorical columns
        :param log: input DataFrame ``[user_id(x), item_id(x), timestamp, relevance]``
        :param features_df: DataFrame with user or item features
        :return: dictionary "categorical feature name - DataFrame with popularity by id and category values"
        """
        if self.item_id in features_df.columns:
            join_col, agg_col = self.item_id, self.user_id
        else:
            join_col, agg_col = self.user_id, self.item_id

        conditional_dist = dict()
        log_with_features = log.join(features_df, on=join_col, how="left")
        count_by_agg_col_name = "count_by_{}".format(agg_col)
        count_by_agg_col = log_with_features.groupBy(agg_col).agg(
            sf.count("relevance").alias(count_by_agg_col_name)
        )
        for cat_col in cat_cols:
            col_name = "{}_pop_by_{}".format(agg_col[:4], cat_col)
            intermediate_df = log_with_features.groupBy(agg_col, cat_col).agg(
                sf.count("relevance").alias(col_name)
            )
            intermediate_df = intermediate_df.join(
                count_by_agg_col, on=agg_col, how="left"
            )
            conditional_dist[cat_col] = intermediate_df.withColumn(
                col_name, sf.col(col_name) / sf.col(count_by_agg_col_name)
            ).drop(count_by_agg_col_name)
            conditional_dist[cat_col].cache()

        return conditional_dist

    def fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
    ) -> None:
        """
        Calculate features for users and items, and popularity based on categorical features.

        :param log: input DataFrame ``[user_id(x), item_id(x), timestamp, relevance]``
        :param user_features: DataFrame with ``user_id(x)`` and feature columns
        :param item_features: DataFrame with ``item_id(x)`` and feature columns
        :param user_cat_features_list: list of user categorical features used to calculate item popularity features,
            such as movie popularity among certain age group
        :param item_cat_features_list: list of item categorical features
        """
        if self.use_cooccurrence:
            raise NotImplementedError("co-occurrence will be implemented soon")

        if self.use_log_features:
            (
                self.user_log_features,
                self.item_log_features,
            ) = self._calc_log_features(log)

        if self.use_conditional_popularity:
            if (
                user_features is not None
                and user_cat_features_list is not None
            ):
                self.item_cond_dist_cat_features = self._add_cond_distr_feat(
                    user_cat_features_list, log, user_features
                )

            if (
                item_features is not None
                and item_cat_features_list is not None
            ):
                self.user_cond_dist_cat_features = self._add_cond_distr_feat(
                    item_cat_features_list, log, item_features
                )

        self.fitted = True

    def transform(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ):
        """
        Add features
        :param log: input DataFrame ``[user_id(x), item_id(x), ...]``
        :param user_features: user features
        :param item_features: item features
        :return: augmented DataFrame
        """
        if not self.fitted:
            raise AttributeError("Call fit before running transform")
        joined = join_or_return(
            log, user_features, how="left", on=self.user_id
        )
        joined = join_or_return(
            joined, item_features, how="left", on=self.item_id
        )

        if self.use_log_features:
            joined = (
                joined.join(
                    self.user_log_features, on=self.user_id, how="left"
                )
                .join(self.item_log_features, on=self.item_id, how="left")
                .fillna(
                    {
                        col_name: 0
                        for col_name in self.user_log_features.columns
                        + self.item_log_features.columns
                    }
                )
            )

            joined = joined.withColumn(
                "u_log_ratings_count_diff",
                sf.col("u_log_ratings_count")
                - sf.col("i_mean_log_users_ratings_count"),
            ).withColumn(
                "i_log_ratings_count_diff",
                sf.col("i_log_ratings_count")
                - sf.col("u_mean_log_items_ratings_count"),
            )

        if self.use_conditional_popularity:
            if self.user_cond_dist_cat_features is not None:
                for key, value in self.user_cond_dist_cat_features.items():
                    joined = join_or_return(
                        joined, value, on=[self.user_id, key], how="left"
                    )
                    joined = joined.fillna({"user_pop_by_" + key: 0})

            if self.item_cond_dist_cat_features is not None:
                for key, value in self.item_cond_dist_cat_features.items():
                    joined = join_or_return(
                        joined, value, on=[self.item_id, key], how="left"
                    )
                    joined = joined.fillna({"item_pop_by_" + key: 0})

        return joined

    def __del__(self):
        for log_feature in [self.user_log_features, self.item_log_features]:
            if log_feature is not None:
                log_feature.unpersist()

        for conditional_feature in [
            self.user_cond_dist_cat_features,
            self.item_cond_dist_cat_features,
        ]:
            if conditional_feature is not None:
                for value in conditional_feature.values():
                    value.unpersist()
