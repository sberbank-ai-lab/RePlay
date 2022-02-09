from typing import Dict, Optional, List

import pyspark.sql.functions as sf

from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, TimestampType

from replay.data_preparator import CatFeaturesTransformer
from replay.session_handler import State
from replay.utils import join_or_return, ugly_join, unpersist_if_exists


class AllToNumericFeatureTransformer:
    """Transform user/item features to numeric types:
    - numeric features stays as is
    - categorical features:
        if threshold is defined:
            - all non-numeric columns with less unique values than threshold are one-hot encoded
            - remaining columns are dropped
        else all non-numeric columns are one-hot encoded
    """

    cat_feat_transformer: Optional[CatFeaturesTransformer]
    cols_to_ohe: Optional[List]
    cols_to_del: Optional[List]
    all_columns: Optional[List]

    def __init__(self, threshold: Optional[int] = 100):
        self.threshold = threshold
        self.fitted = False

    def fit(self, features: Optional[DataFrame]) -> None:
        """
        Determine categorical columns for one-hot encoding.
        Non categorical columns with more values than threshold will be deleted.
        Saves categories for each column.
        :param features: input DataFrame
        """
        self.cat_feat_transformer = None
        self.cols_to_del = []
        self.fitted = True

        if features is None:
            self.all_columns = None
            return

        self.all_columns = sorted(features.columns)
        idx_cols_set = {"user_idx", "item_idx", "user_id", "item_id"}

        spark_df_non_numeric_cols = [
            col
            for col in features.columns
            if (not isinstance(features.schema[col].dataType, NumericType))
            and (col not in idx_cols_set)
        ]

        # numeric only
        if len(spark_df_non_numeric_cols) == 0:
            self.cols_to_ohe = []
            return

        if self.threshold is None:
            self.cols_to_ohe = spark_df_non_numeric_cols
        else:
            counts_pd = (
                features.agg(
                    *[
                        sf.approx_count_distinct(sf.col(c)).alias(c)
                        for c in spark_df_non_numeric_cols
                    ]
                )
                .toPandas()
                .T
            )
            self.cols_to_ohe = (
                counts_pd[counts_pd[0] <= self.threshold]
            ).index.values

            self.cols_to_del = [
                col
                for col in spark_df_non_numeric_cols
                if col not in set(self.cols_to_ohe)
            ]

            if self.cols_to_del:
                State().logger.warning(
                    "%s columns contain more that threshold unique "
                    "values and will be deleted",
                    self.cols_to_del,
                )

        if len(self.cols_to_ohe) > 0:
            self.cat_feat_transformer = CatFeaturesTransformer(
                cat_cols_list=self.cols_to_ohe
            )
            self.cat_feat_transformer.fit(features.drop(*self.cols_to_del))

    def transform(self, spark_df: Optional[DataFrame]) -> Optional[DataFrame]:
        """
        Transform categorical features.
        Use one hot encoding for columns with the amount of unique values smaller
        than threshold and delete other columns.
        :param spark_df: input DataFrame
        :return: processed DataFrame
        """
        if not self.fitted:
            raise AttributeError("Call fit before running transform")

        if spark_df is None or self.all_columns is None:
            return None

        if self.cat_feat_transformer is None:
            return spark_df.drop(*self.cols_to_del)

        if sorted(spark_df.columns) != self.all_columns:
            raise ValueError(
                f"Columns from fit do not match "
                f"columns in transform. "
                f"Fit columns: {self.all_columns},"
                f"Transform columns: {spark_df.columns}"
            )

        return self.cat_feat_transformer.transform(
            spark_df.drop(*self.cols_to_del)
        )

    def fit_transform(self, spark_df: DataFrame) -> DataFrame:
        """
        :param spark_df: input DataFrame
        :return: output DataFrame
        """
        self.fit(spark_df)
        return self.transform(spark_df)


class EmptyFeatureProcessor:
    """Do not perform any transformations on the dataframe"""

    @staticmethod
    def fit(log: DataFrame, features: DataFrame) -> None:
        """
        :param log: input DataFrame ``[user_idx, item_idx, timestamp, relevance]``
        :param features: DataFrame with ``user_idx/item_idx`` and feature columns
        """

    @staticmethod
    def transform(log: DataFrame) -> DataFrame:
        """
        Return log without any transformations
        :param log: spark DataFrame
        """
        return log


class LogStatFeaturesProcessor(EmptyFeatureProcessor):
    """
    Calculate user and item features based on interactions log:
        Based on the number of interactions:
        - log number of interactions (1)
        - average log number of interactions by users interacted with item and vice versa (2)
        - difference between number of interactions by user/item (1)
        and average number of interactions (2)

        Based on timestamp (if present and has a TimestampType):
        - min and max interaction timestamp for user/item
        - history length (max - min timestamp)
        - log number of interactions' days
        - difference in days between last date in log and last interaction of the user/item

        Based or ratings/relevance:
        - relevance mean and std
        - relevance approximate quantiles (0.05, 0.5, 0.95)
        - abnormality of user's preferences https://hal.inria.fr/hal-01254172/document
    """

    calc_timestamp_based: bool = False
    calc_relevance_based: bool = False
    user_log_features: Optional[DataFrame] = None
    item_log_features: Optional[DataFrame] = None

    def _create_log_aggregates(self, agg_col: str = "user_idx") -> List:
        """
        Create features based on relevance type
        (binary or not) and whether timestamp is present.
        :param agg_col: column to create features for, user_idx or item_idx
        :return: list of columns to pass into pyspark agg
        """
        prefix = agg_col[:1]

        aggregates = [
            sf.log(sf.count(sf.col("relevance"))).alias(
                f"{prefix}_log_num_interact"
            )
        ]

        if self.calc_timestamp_based:
            aggregates.extend(
                [
                    sf.log(
                        sf.countDistinct(
                            sf.date_trunc("dd", sf.col("timestamp"))
                        )
                    ).alias(f"{prefix}_log_interact_days_count"),
                    sf.min(sf.col("timestamp")).alias(
                        f"{prefix}_min_interact_date"
                    ),
                    sf.max(sf.col("timestamp")).alias(
                        f"{prefix}_max_interact_date"
                    ),
                ]
            )

        if self.calc_relevance_based:
            aggregates.extend(
                [
                    (
                        sf.when(
                            sf.stddev(sf.col("relevance")).isNull()
                            | sf.isnan(sf.stddev(sf.col("relevance"))),
                            0,
                        )
                        .otherwise(sf.stddev(sf.col("relevance")))
                        .alias(f"{prefix}_std")
                    ),
                    sf.mean(sf.col("relevance")).alias(f"{prefix}_mean"),
                ]
            )
            for percentile in [0.05, 0.5, 0.95]:
                aggregates.append(
                    sf.expr(
                        f"percentile_approx(relevance, {percentile})"
                    ).alias(f"{prefix}_quantile_{str(percentile)[2:]}")
                )

        return aggregates

    @staticmethod
    def _add_ts_based(
        features: DataFrame, max_log_date: datetime, prefix: str
    ) -> DataFrame:
        """
        Add history length (max - min timestamp) and difference in days between
        last date in log and last interaction of the user/item

        :param features: dataframe with calculated log-based features
        :param max_log_date: max timestamp in log used for features calculation
        :param prefix: identifier used as a part of column name
        :return: features dataframe with new timestamp-based columns
        """
        return features.withColumn(
            f"{prefix}_history_length_days",
            sf.datediff(
                sf.col(f"{prefix}_max_interact_date"),
                sf.col(f"{prefix}_min_interact_date"),
            ),
        ).withColumn(
            f"{prefix}_last_interaction_gap_days",
            sf.datediff(
                sf.lit(max_log_date), sf.col(f"{prefix}_max_interact_date")
            ),
        )

    @staticmethod
    def _cals_cross_interactions_count(
        log: DataFrame, features: DataFrame
    ) -> DataFrame:
        """
        Calculate difference between the log number of interactions by the user
        and average log number of interactions users interacted with the item has.

        :param log: dataframe with calculated log-based features
        :param features: dataframe with calculated log-based features
        :return: features dataframe with new columns
        """
        if "user_idx" in features.columns:
            new_feature_entity, calc_by_entity = "item_idx", "user_idx"
        else:
            new_feature_entity, calc_by_entity = "user_idx", "item_idx"

        mean_log_num_interact = log.join(
            features.select(
                calc_by_entity, f"{calc_by_entity[0]}_log_num_interact"
            ),
            on=calc_by_entity,
            how="left",
        )
        return mean_log_num_interact.groupBy(new_feature_entity).agg(
            sf.mean(f"{calc_by_entity[0]}_log_num_interact").alias(
                f"{new_feature_entity[0]}_mean_{calc_by_entity[0]}_log_num_interact"
            )
        )

    @staticmethod
    def _calc_abnormality(
        log: DataFrame, item_features: DataFrame
    ) -> DataFrame:
        """
        Calculate  discrepancy between a rating on a resource
        and the average rating of this resource (Abnormality) and
        abnormality taking controversy of the item into account (AbnormalityCR).
        https://hal.inria.fr/hal-01254172/document

        :param log: dataframe with calculated log-based features
        :param item_features: dataframe with calculated log-based features
        :return: features dataframe with new columns
        """
        # Abnormality
        abnormality_df = ugly_join(
            left=log,
            right=item_features.select("item_idx", "i_mean", "i_std"),
            on_col_name="item_idx",
            how="left",
        )
        abnormality_df = abnormality_df.withColumn(
            "abnormality", sf.abs(sf.col("relevance") - sf.col("i_mean"))
        )

        abnormality_aggs = [
            sf.mean(sf.col("abnormality")).alias("abnormality")
        ]

        # Abnormality CR:
        max_std = item_features.select(sf.max("i_std")).collect()[0][0]
        min_std = item_features.select(sf.min("i_std")).collect()[0][0]
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

        return abnormality_df.groupBy("user_idx").agg(*abnormality_aggs)

    def fit(
        self, log: DataFrame, features: Optional[DataFrame] = None
    ) -> None:
        """
        Calculate log-based features for users and items

         :param log: input DataFrame ``[user_idx, item_idx, timestamp, relevance]``
         :param features: not required
        """
        self.calc_timestamp_based = (
            isinstance(log.schema["timestamp"].dataType, TimestampType)
        ) & (
            log.select(sf.countDistinct(sf.col("timestamp"))).collect()[0][0]
            > 1
        )
        self.calc_relevance_based = (
            log.select(sf.countDistinct(sf.col("relevance"))).collect()[0][0]
            > 1
        )

        user_log_features = log.groupBy("user_idx").agg(
            *self._create_log_aggregates(agg_col="user_idx")
        )
        item_log_features = log.groupBy("item_idx").agg(
            *self._create_log_aggregates(agg_col="item_idx")
        )

        if self.calc_timestamp_based:
            last_date = log.select(sf.max("timestamp")).collect()[0][0]
            user_log_features = self._add_ts_based(
                features=user_log_features, max_log_date=last_date, prefix="u"
            )

            item_log_features = self._add_ts_based(
                features=item_log_features, max_log_date=last_date, prefix="i"
            )

        if self.calc_relevance_based:
            user_log_features = user_log_features.join(
                self._calc_abnormality(
                    log=log, item_features=item_log_features
                ),
                on="user_idx",
                how="left",
            ).cache()

        self.user_log_features = ugly_join(
            left=user_log_features,
            right=self._cals_cross_interactions_count(
                log=log, features=item_log_features
            ),
            on_col_name="user_idx",
            how="left",
        ).cache()

        self.item_log_features = ugly_join(
            left=item_log_features,
            right=self._cals_cross_interactions_count(
                log=log, features=user_log_features
            ),
            on_col_name="item_idx",
            how="left",
        ).cache()

    def transform(self, log: DataFrame) -> DataFrame:
        """
        Add log-based features for users and items

        :param log: input DataFrame with
            ``[user_idx, item_idx, <features columns>]`` columns
        :return: log with log-based feature columns
        """
        joined = (
            log.join(
                self.user_log_features,
                on="user_idx",
                how="left",
            )
            .join(
                self.item_log_features,
                on="item_idx",
                how="left",
            )
            .withColumn(
                "na_u_log_features",
                sf.when(sf.col("u_log_num_interact").isNull(), 1.0).otherwise(
                    0.0
                ),
            )
            .withColumn(
                "na_i_log_features",
                sf.when(sf.col("i_log_num_interact").isNull(), 1.0).otherwise(
                    0.0
                ),
            )
            # TO DO std и date diff заменяем на inf, date features - будут ли работать корректно?
            # если не заменять, будет ли работать корректно?
            .fillna(
                {
                    col_name: 0
                    for col_name in self.user_log_features.columns
                    + self.item_log_features.columns
                }
            )
        )

        joined = joined.withColumn(
            "u_i_log_num_interact_diff",
            sf.col("u_log_num_interact")
            - sf.col("i_mean_log_users_interact_count"),
        ).withColumn(
            "i_u_log_num_interact_diff",
            sf.col("i_log_num_interact")
            - sf.col("u_mean_log_items_interact_count"),
        )

        return joined

    def __del__(self):
        unpersist_if_exists(self.user_log_features)
        unpersist_if_exists(self.item_log_features)


class ConditionalPopularityProcessor(EmptyFeatureProcessor):
    """
    Calculate popularity based on user or item categorical features
    (for example movie popularity among users of the same age group).
    If user features are provided, item features will be generated and vice versa.
    """

    conditional_pop_dict: Optional[Dict[str, DataFrame]]
    entity_name: str

    def __init__(
        self,
        cat_features_list: List,
    ):
        """
        :param cat_features_list: List of columns with categorical features to use
            for conditional popularity calculation
        """
        self.cat_features_list = cat_features_list

    def fit(self, log: DataFrame, features: DataFrame) -> None:
        """
        Calculate conditional popularity for id and categorical features
        defined in `cat_features_list`

        :param log: input DataFrame ``[user_idx, item_idx, timestamp, relevance]``
        :param features: DataFrame with ``user_idx/item_idx`` and feature columns
        """
        if len(
            set(self.cat_features_list).intersection(features.columns)
        ) != len(self.cat_features_list):
            raise ValueError(
                f"Columns {set(self.cat_features_list).difference(features.columns)} "
                f"defined in `cat_features_list` are absent in features. "
                f"features columns are: {features.columns}."
            )

        join_col, self.entity_name = (
            ("item_idx", "user_idx")
            if "item_idx" in features.columns
            else ("user_idx", "item_idx")
        )

        self.conditional_pop_dict = {}
        log_with_features = log.join(features, on=join_col, how="left")
        count_by_entity_col_name = f"count_by_{self.entity_name}"

        count_by_entity_col = log_with_features.groupBy(self.entity_name).agg(
            sf.count("relevance").alias(count_by_entity_col_name)
        )

        for cat_col in self.cat_features_list:
            col_name = f"{self.entity_name[0]}_pop_by_{cat_col}"
            intermediate_df = log_with_features.groupBy(
                self.entity_name, cat_col
            ).agg(sf.count("relevance").alias(col_name))
            intermediate_df = intermediate_df.join(
                sf.broadcast(count_by_entity_col),
                on=self.entity_name,
                how="left",
            )
            self.conditional_pop_dict[cat_col] = intermediate_df.withColumn(
                col_name, sf.col(col_name) / sf.col(count_by_entity_col_name)
            ).drop(count_by_entity_col_name)
            self.conditional_pop_dict[cat_col].cache()

    def transform(self, log: DataFrame) -> DataFrame:
        """
        Add conditional popularity features

        :param log: input DataFrame with
            ``[user_idx, item_idx, <features columns>]`` columns
        :return: log with conditional popularity feature columns
        """

        joined = log
        for (
            key,
            value,
        ) in self.conditional_pop_dict.items():
            joined = join_or_return(
                joined,
                sf.broadcast(value),
                on=["user_idx", key],
                how="left",
            )
            joined = joined.fillna({"user_pop_by_" + key: 0})
        return joined

    def __del__(self):
        for df in self.conditional_pop_dict.values():
            unpersist_if_exists(df)


# pylint: disable=too-many-instance-attributes, too-many-arguments
class HistoryBasedFeaturesProcessor:
    """
    Calculate user and item features based on interactions history (log).
    calculated features includes numbers of interactions, rating and timestamp distribution features
    and conditional popularity for pairs `user_idx/item_idx - categorical feature`.

    See LogStatFeaturesProcessor and ConditionalPopularityProcessor documentation
    for detailed description of generated features.
    """

    log_proc = EmptyFeatureProcessor()
    user_cond_pop_proc = EmptyFeatureProcessor()
    item_cond_pop_proc = EmptyFeatureProcessor()

    def __init__(
        self,
        use_log_features: bool = True,
        use_conditional_popularity: bool = True,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
    ):
        """
        :param use_log_features: if add statistical log-based features
            generated by LogStatFeaturesProcessor
        :param use_conditional_popularity: if add conditional popularity
            features generated by ConditionalPopularityProcessor
        :param user_cat_features_list: list of user categorical features
            used to calculate item conditional popularity features
        :param item_cat_features_list: list of item categorical features
            used to calculate user conditional popularity features
        """
        if use_log_features:
            self.log_processor = LogStatFeaturesProcessor()
        if use_conditional_popularity and user_cat_features_list:
            if user_cat_features_list:
                self.user_cond_pop_proc = ConditionalPopularityProcessor(
                    cat_features_list=user_cat_features_list
                )
            if item_cat_features_list:
                self.item_cond_pop_proc = ConditionalPopularityProcessor(
                    cat_features_list=item_cat_features_list
                )
        self.fitted: bool = False

    def fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Calculate log and conditional popularity features.

        :param log: input DataFrame ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: DataFrame with ``user_idx`` and feature columns
        :param item_features: DataFrame with ``item_idx`` and feature columns
        """
        log = log.cache()
        self.log_processor.fit(log=log)
        self.user_cond_pop_proc.fit(log=log, features=item_features)
        self.item_cond_pop_proc.fit(log=log, features=user_features)
        self.fitted = True
        log.unpersist()

    def transform(
        self,
        log: DataFrame,
    ):
        """
        Add features
        :param log: input DataFrame with
            ``[user_idx, item_idx, <features columns>]`` columns
        :return: augmented DataFrame
        """
        if not self.fitted:
            raise AttributeError("Call fit before running transform")
        joined = self.log_processor.transform(log)
        joined = self.user_cond_pop_proc.transform(joined)
        joined = self.item_cond_pop_proc.transform(joined)

        return joined
