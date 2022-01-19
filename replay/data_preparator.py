"""
Contains classes ``DataPreparator`` and ``CatFeaturesTransformer``.
``DataPreparator`` is used to transform DataFrames to a library format.

`CatFeaturesTransformer`` transforms categorical features with one-hot encoding.
"""
import string
from typing import Dict, List, Optional

from pyspark.ml.feature import StringIndexerModel, IndexToString, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf


from replay.constants import AnyDataFrame
from replay.utils import convert2spark


class Indexer:
    """
    This class is used to convert arbitrary id to numerical idx and back.
    """

    user_indexer: StringIndexerModel
    item_indexer: StringIndexerModel
    inv_user_indexer: IndexToString
    inv_item_indexer: IndexToString
    user_type: None
    item_type: None

    def fit(
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
        self.user_type = log.schema["user_id"].dataType
        self.item_type = log.schema["item_id"].dataType

    def update(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Update indexers with new ids if there are any.

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features (must have ``user_id``)
        :param item_features: item features (must have ``item_id``)
        :return:
        """
        for df in [log, user_features, item_features]:
            self._reindex(df, "user")
            self._reindex(df, "item")

    def transform(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> tuple:
        """
        Convert ids into idxs for provided DataFrames

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features (must have ``user_id``)
        :param item_features: item features (must have ``item_id``)
        :return: three converted DataFrames
        """
        log_i = self._transform(log)
        user_features_i = self._transform(user_features)
        item_features_i = self._transform(item_features)
        return log_i, user_features_i, item_features_i

    def fit_transform(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> tuple:
        """
        Creates indexers and convert provided DataFrames

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features (must have ``user_id``)
        :param item_features: item features (must have ``item_id``)
        :return: three converted DataFrames
        """
        self.fit(log, user_features, item_features)
        return self.transform(log, user_features, item_features)

    def _transform(
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
            self._reindex(data_frame, "user")
            data_frame = self.user_indexer.transform(data_frame).drop(
                "user_id"
            )
            data_frame = data_frame.withColumn(
                "user_idx", sf.col("user_idx").cast("int")
            )
        if "item_id" in data_frame.columns:
            self._reindex(data_frame, "item")
            data_frame = self.item_indexer.transform(data_frame).drop(
                "item_id"
            )
            data_frame = data_frame.withColumn(
                "item_idx", sf.col("item_idx").cast("int")
            )
        return data_frame

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Convert DataFrame to the initial indexes.

        :param df: DataFrame with idxs
        :return: DataFrame with ids
        """
        res = df
        if "user_idx" in df.columns:
            res = (
                self.inv_user_indexer.transform(res)
                .drop("user_idx")
                .withColumn("user_id", sf.col("user_id").cast(self.user_type))
            )
        if "item_idx" in df.columns:
            res = (
                self.inv_item_indexer.transform(res)
                .drop("item_idx")
                .withColumn("item_id", sf.col("item_id").cast(self.item_type))
            )
        return res

    def _reindex(self, df: DataFrame, entity: str):
        """
        Update indexer with new entries.

        :param df: DataFrame with users/items
        :param entity: user or item
        """
        indexer = getattr(self, f"{entity}_indexer")
        inv_indexer = getattr(self, f"inv_{entity}_indexer")
        new_objects = set(
            map(
                str,
                df.select(sf.collect_list(indexer.getInputCol())).first()[0],
            )
        ).difference(indexer.labels)
        if new_objects:
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


class CatFeaturesTransformer:
    """Transform categorical features in ``cat_cols_list``
    with one-hot encoding and delete other columns."""

    def __init__(
        self, cat_cols_list: List, alias: str = "ohe",
    ):
        """
        :param cat_cols_list: list of categorical columns
        :param alias: prefix for one-hot encoding columns
        """
        self.cat_cols_list = cat_cols_list
        self.expressions_list = []
        self.alias = alias

    def fit(self, spark_df: Optional[DataFrame]) -> None:
        """
        Save categories for each column
        :param spark_df: Spark DataFrame with features
        """
        if spark_df is None:
            return

        cat_feat_values_dict = {
            name: (
                spark_df.select(sf.collect_set(sf.col(name))).collect()[0][0]
            )
            for name in self.cat_cols_list
        }
        self.expressions_list = [
            sf.when(sf.col(col_name) == cur_name, 1)
            .otherwise(0)
            .alias(
                f"""{self.alias}_{col_name}_{str(cur_name).translate(
                        str.maketrans(
                            "", "", string.punctuation + string.whitespace
                        )
                    )[:30]}"""
            )
            for col_name, col_values in cat_feat_values_dict.items()
            for cur_name in col_values
        ]

    def transform(self, spark_df: Optional[DataFrame]):
        """
        Transform categorical columns.
        If there are any new categories that were not present at fit stage, they will be ignored.
        :param spark_df: feature DataFrame
        :return: transformed DataFrame
        """
        if spark_df is None:
            return None
        return spark_df.select(*spark_df.columns, *self.expressions_list).drop(
            *self.cat_cols_list
        )


class DataPreparator:
    """
    Convert pandas DataFrame to Spark, rename columns and apply indexer.
    """

    def __init__(self):
        self.indexer = Indexer()

    def __call__(
        self,
        log: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        mapping: Optional[Dict] = None,
    ) -> tuple:
        """
        Convert ids into idxs for provided DataFrames

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features (must have ``user_id``)
        :param item_features: item features (must have ``item_id``)
        :param mapping: dictionary mapping "default column name:
        column name in input DataFrame"
        ``user_id`` and ``item_id`` mappings are required,
        ``timestamp`` and``relevance`` are optional.
        :return: three converted DataFrames
        """
        log, user_features, item_features = [
            convert2spark(df) for df in [log, user_features, item_features]
        ]
        log, user_features, item_features = [
            self._rename(df, mapping)
            for df in [log, user_features, item_features]
        ]
        return self.indexer.fit_transform(log, user_features, item_features)

    @staticmethod
    def _rename(df: DataFrame, mapping: Dict) -> DataFrame:
        if df is None or mapping is None:
            return df
        for out_col, in_col in mapping.items():
            if in_col in df.columns:
                df = df.withColumnRenamed(in_col, out_col)
        return df

    def back(self, df: DataFrame) -> DataFrame:
        """
        Convert DataFrame to the initial indexes.

        :param df: DataFrame with idxs
        :return: DataFrame with ids
        """
        return self.indexer.inverse_transform(df)
