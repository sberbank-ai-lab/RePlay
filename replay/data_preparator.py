"""
Contains classes for data preparation and categorical features transformation.
``DataPreparator`` is used to transform DataFrames to a library format.
``Indexed`` is used to convert user and item ids to numeric format.
``CatFeaturesTransformer`` transforms categorical features with one-hot encoding.
``ToNumericFeatureTransformer`` leaves only numerical features
by one-hot encoding of some features and deleting the others.
"""
import logging
import string
from typing import Dict, List, Optional

from pyspark.ml.feature import StringIndexerModel, IndexToString, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, NumericType

from replay.constants import AnyDataFrame
from replay.session_handler import State
from replay.utils import convert2spark, process_timestamp_column

LOG_COLUMNS = ["user_id", "item_id", "timestamp", "relevance"]


class Indexer:  # pylint: disable=too-many-instance-attributes
    """
    This class is used to convert arbitrary id to numerical idx and back.
    """

    user_indexer: StringIndexerModel
    item_indexer: StringIndexerModel
    inv_user_indexer: IndexToString
    inv_item_indexer: IndexToString
    user_type: None
    item_type: None
    suffix = "inner"

    def __init__(self, user_col="user_id", item_col="item_id"):
        """
        Provide column names for indexer to use
        """
        self.user_col = user_col
        self.item_col = item_col

    @property
    def _init_args(self):
        return {
            "user_col": self.user_col,
            "item_col": self.item_col,
        }

    def fit(
        self,
        users: DataFrame,
        items: DataFrame,
    ) -> None:
        """
        Creates indexers to map raw id to numerical idx so that spark can handle them.
        :param users: DataFrame containing user column
        :param items: DataFrame containing item column
        :return:
        """
        users = users.select(self.user_col).withColumnRenamed(
            self.user_col, f"{self.user_col}_{self.suffix}"
        )
        items = items.select(self.item_col).withColumnRenamed(
            self.item_col, f"{self.item_col}_{self.suffix}"
        )

        self.user_type = users.schema[
            f"{self.user_col}_{self.suffix}"
        ].dataType
        self.item_type = items.schema[
            f"{self.item_col}_{self.suffix}"
        ].dataType

        self.user_indexer = StringIndexer(
            inputCol=f"{self.user_col}_{self.suffix}", outputCol="user_idx"
        ).fit(users)
        self.item_indexer = StringIndexer(
            inputCol=f"{self.item_col}_{self.suffix}", outputCol="item_idx"
        ).fit(items)
        self.inv_user_indexer = IndexToString(
            inputCol=f"{self.user_col}_{self.suffix}",
            outputCol=self.user_col,
            labels=self.user_indexer.labels,
        )
        self.inv_item_indexer = IndexToString(
            inputCol=f"{self.item_col}_{self.suffix}",
            outputCol=self.item_col,
            labels=self.item_indexer.labels,
        )

    def transform(self, df: DataFrame) -> Optional[DataFrame]:
        """
        Convert raw ``user_col`` and ``item_col`` to numerical ``user_idx`` and ``item_idx``

        :param df: dataframe with raw indexes
        :return: dataframe with converted indexes
        """
        if self.user_col in df.columns:
            df = df.withColumnRenamed(
                self.user_col, f"{self.user_col}_{self.suffix}"
            )
            self._reindex(df, "user")
            df = self.user_indexer.transform(df).drop(
                f"{self.user_col}_{self.suffix}"
            )
            df = df.withColumn("user_idx", sf.col("user_idx").cast("int"))
        if self.item_col in df.columns:
            df = df.withColumnRenamed(
                self.item_col, f"{self.item_col}_{self.suffix}"
            )
            self._reindex(df, "item")
            df = self.item_indexer.transform(df).drop(
                f"{self.item_col}_{self.suffix}"
            )
            df = df.withColumn("item_idx", sf.col("item_idx").cast("int"))
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Convert DataFrame to the initial indexes.

        :param df: DataFrame with numerical ``user_idx/item_idx`` columns
        :return: DataFrame with original user/item columns
        """
        res = df
        if "user_idx" in df.columns:
            res = (
                self.inv_user_indexer.transform(
                    res.withColumnRenamed(
                        "user_idx", f"{self.user_col}_{self.suffix}"
                    )
                )
                .drop(f"{self.user_col}_{self.suffix}")
                .withColumn(
                    self.user_col, sf.col(self.user_col).cast(self.user_type)
                )
            )
        if "item_idx" in df.columns:
            res = (
                self.inv_item_indexer.transform(
                    res.withColumnRenamed(
                        "item_idx", f"{self.item_col}_{self.suffix}"
                    )
                )
                .drop(f"{self.item_col}_{self.suffix}")
                .withColumn(
                    self.item_col, sf.col(self.item_col).cast(self.item_type)
                )
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


class DataPreparator:
    """Transforms data to a library format:
        - read as a spark dataframe/ convert pandas dataframe to spark
        - check for nulls
        - create relevance/timestamp columns if absent
        - convert dates to TimestampType

    Examples:

    Loading log DataFrame

    >>> import pandas as pd
    >>> from replay.data_preparator import DataPreparator
    >>>
    >>> log = pd.DataFrame({"user": [2, 2, 2, 1],
    ...                     "item_id": [1, 2, 3, 3],
    ...                     "rel": [5, 5, 5, 5]}
    ...                    )
    >>> dp = DataPreparator()
    >>> correct_log = dp.transform(data=log,
    ...                            columns_mapping={"user_id": "user",
    ...                                           "item_id": "item_id",
    ...                                           "relevance": "rel"}
    ...                             )
    >>> correct_log.show(2)
    +-------+-------+---------+-------------------+
    |user_id|item_id|relevance|          timestamp|
    +-------+-------+---------+-------------------+
    |      2|      1|      5.0|2099-01-01 00:00:00|
    |      2|      2|      5.0|2099-01-01 00:00:00|
    +-------+-------+---------+-------------------+
    only showing top 2 rows
    <BLANKLINE>


    Loading user features

    >>> import pandas as pd
    >>> from replay.data_preparator import DataPreparator
    >>>
    >>> log = pd.DataFrame({"user": ["user1", "user1", "user2"],
    ...                     "f0": ["feature1","feature2","feature1"],
    ...                     "f1": ["left","left","center"],
    ...                     "ts": ["2019-01-01","2019-01-01","2019-01-01"]}
    ...             )
    >>> dp = DataPreparator()
    >>> correct_log = dp.transform(data=log,
    ...                            columns_mapping={"user_id": "user"},
    ...                             )
    >>> correct_log.show(3)
    +-------+--------+------+----------+
    |user_id|      f0|    f1|        ts|
    +-------+--------+------+----------+
    |  user1|feature1|  left|2019-01-01|
    |  user1|feature2|  left|2019-01-01|
    |  user2|feature1|center|2019-01-01|
    +-------+--------+------+----------+
    <BLANKLINE>

    """

    _logger: Optional[logging.Logger] = None

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    @staticmethod
    def read_as_spark_df(
        data: Optional[AnyDataFrame] = None,
        path: str = None,
        format_type: str = None,
        **kwargs,
    ) -> DataFrame:
        """
        Read spark dataframe from file of transform pandas dataframe.

        :param data: DataFrame to process (``pass`` or ``data`` should be defined)
        :param path: path to data (``pass`` or ``data`` should be defined)
        :param format_type: file type, one of ``[csv , parquet , json , table]``
        :param kwargs: extra arguments passed to
            ``spark.read.<format>(path, **reader_kwargs)``
        :return: spark DataFrame
        """
        if data is not None:
            dataframe = convert2spark(data)
        elif path and format_type:
            spark = State().session
            if format_type == "csv":
                dataframe = spark.read.csv(path, inferSchema=True, **kwargs)
            elif format_type == "parquet":
                dataframe = spark.read.parquet(path)
            elif format_type == "json":
                dataframe = spark.read.json(path, **kwargs)
            elif format_type == "table":
                dataframe = spark.read.table(path)
            else:
                raise ValueError(
                    f"Invalid value of format_type='{format_type}'"
                )
        else:
            raise ValueError("Either data or path parameters must not be None")
        return dataframe

    def check_df(
        self, dataframe: DataFrame, columns_mapping: Dict[str, str]
    ) -> None:
        """
        Check:
        - if dataframe is not empty,
        - if columns from ``columns_mapping`` are present in dataframe
        - warn about nulls in columns from ``columns_mapping``
        - warn about absent of ``timestamp/relevance`` columns for interactions log
        - warn about wrong relevance DataType

        :param dataframe: spark DataFrame to process
        :param columns_mapping: dictionary mapping "key: column name in input DataFrame".
            Possible keys: ``[user_id, user_id, timestamp, relevance]``
            ``columns_mapping`` values specifies the nature of the DataFrame:
            - if both ``[user_id, item_id]`` are present,
            then the dataframe is a log of interactions.
            Specify ``timestamp, relevance`` columns in mapping if available.
            - if ether ``user_id`` or ``item_id`` is present,
            then the dataframe is a dataframe of user/item features
        """
        if not dataframe.head(1):
            raise ValueError("DataFrame is empty")

        for value in columns_mapping.values():
            if value not in dataframe.columns:
                raise ValueError(
                    f"Column `{value}` stated in mapping is absent in dataframe"
                )

        for column in columns_mapping.values():
            if dataframe.where(sf.col(column).isNull()).count() > 0:
                self.logger.info(
                    "Column `%s` has NULL values. Handle NULL values before "
                    "the next data preprocessing/model training steps",
                    column,
                )

        if (
            "user_id" in columns_mapping.keys()
            and "item_id" in columns_mapping.keys()
        ):
            absent_cols = set(LOG_COLUMNS).difference(columns_mapping.keys())
            if len(absent_cols) > 0:
                self.logger.info(
                    "Columns %s are absent, but may be required for models training. "
                    "Add them with DataPreparator().generate_absent_log_cols",
                    list(absent_cols),
                )
        if "relevance" in columns_mapping.keys():
            if not isinstance(
                dataframe.schema[columns_mapping["relevance"]].dataType,
                NumericType,
            ):
                self.logger.info(
                    "Relevance column `%s` should be numeric, " "but it is %s",
                    columns_mapping["relevance"],
                    dataframe.schema[columns_mapping["relevance"]].dataType,
                )

    @staticmethod
    def add_absent_log_cols(
        dataframe: DataFrame,
        columns_mapping: Dict[str, str],
        default_relevance: float = 1.0,
        default_ts: str = "2099-01-01",
    ):
        """
        Add ``relevance`` and ``timestamp`` columns with default values if
         ``relevance`` or ``timestamp`` is absent among mapping keys.

        :param dataframe: interactions log to process
        :param columns_mapping: dictionary mapping "key: column name in input DataFrame".
            Possible keys: ``[user_id, user_id, timestamp, relevance]``
        :param default_relevance: default value for generated `relevance` column
        :param default_ts: str, default value for generated `timestamp` column
        :return: spark DataFrame with generated ``timestamp`` and ``relevance`` columns
            if absent in original dataframe
        """
        absent_cols = set(LOG_COLUMNS).difference(columns_mapping.keys())
        if "relevance" in absent_cols:
            dataframe = dataframe.withColumn(
                "relevance", sf.lit(default_relevance).cast(DoubleType())
            )
        if "timestamp" in absent_cols:
            dataframe = dataframe.withColumn(
                "timestamp", sf.to_timestamp(sf.lit(default_ts))
            )
        return dataframe

    @staticmethod
    def _rename(df: DataFrame, mapping: Dict) -> Optional[DataFrame]:
        """
        rename dataframe columns based on mapping
        """
        if df is None or mapping is None:
            return df
        for out_col, in_col in mapping.items():
            if in_col in df.columns:
                df = df.withColumnRenamed(in_col, out_col)
        return df

    # pylint: disable=too-many-arguments
    def transform(
        self,
        columns_mapping: Dict[str, str],
        data: Optional[AnyDataFrame] = None,
        path: Optional[str] = None,
        format_type: Optional[str] = None,
        date_format: Optional[str] = None,
        reader_kwargs: Optional[Dict] = None,
    ) -> DataFrame:
        """
        Transforms log, user or item features into a Spark DataFrame
        ``[user_id, user_id, timestamp, relevance]``,
        ``[user_id, *features]``, or  ``[item_id, *features]``.
        Input is either file of ``format_type``
        at ``path``, or ``pandas.DataFrame`` or ``spark.DataFrame``.
        Transform performs:
        - dataframe reading/convert to spark DataFrame format
        - check dataframe (nulls, columns_mapping)
        - rename columns from mapping to standard names (user_id, user_id, timestamp, relevance)
        - for interactions log: create absent columns,
        convert ``timestamp`` column to TimestampType and ``relevance`` to DoubleType

        :param columns_mapping: dictionary mapping "key: column name in input DataFrame".
            Possible keys: ``[user_id, user_id, timestamp, relevance]``
            ``columns_mapping`` values specifies the nature of the DataFrame:
            - if both ``[user_id, item_id]`` are present,
            then the dataframe is a log of interactions.
            Specify ``timestamp, relevance`` columns in mapping if present.
            - if ether ``user_id`` or ``item_id`` is present,
            then the dataframe is a dataframe of user/item features

        :param data: DataFrame to process
        :param path: path to data
        :param format_type: file type, one of ``[csv , parquet , json , table]``
        :param date_format: format for the ``timestamp`` column
        :param reader_kwargs: extra arguments passed to
            ``spark.read.<format>(path, **reader_kwargs)``
        :return: processed DataFrame
        """
        is_log = False
        if (
            "user_id" in columns_mapping.keys()
            and "item_id" in columns_mapping.keys()
        ):
            self.logger.info(
                "Columns with ids of users or items are present in mapping. "
                "The dataframe will be treated as an interactions log."
            )
            is_log = True
        elif (
            "user_id" not in columns_mapping.keys()
            and "item_id" not in columns_mapping.keys()
        ):
            raise ValueError(
                "Mapping either for user ids or for item ids is not stated in `columns_mapping`"
            )
        else:
            self.logger.info(
                "Column with ids of users or items is absent in mapping. "
                "The dataframe will be treated as a users'/items' features dataframe."
            )
        reader_kwargs = {} if reader_kwargs is None else reader_kwargs
        dataframe = self.read_as_spark_df(
            data=data, path=path, format_type=format_type, **reader_kwargs
        )
        self.check_df(dataframe, columns_mapping=columns_mapping)
        dataframe = self._rename(df=dataframe, mapping=columns_mapping)
        if is_log:
            dataframe = self.add_absent_log_cols(
                dataframe=dataframe, columns_mapping=columns_mapping
            )
            dataframe = dataframe.withColumn(
                "relevance", sf.col("relevance").cast(DoubleType())
            )
            dataframe = process_timestamp_column(
                dataframe=dataframe,
                column_name="timestamp",
                date_format=date_format,
            )

        return dataframe


class CatFeaturesTransformer:
    """Transform categorical features in ``cat_cols_list``
    with one-hot encoding and remove original columns."""

    def __init__(
        self,
        cat_cols_list: List,
        alias: str = "ohe",
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


class ToNumericFeatureTransformer:
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

        spark_df_non_numeric_cols = [
            col
            for col in features.columns
            if (not isinstance(features.schema[col].dataType, NumericType))
            and (col not in {"user_idx", "item_idx"})
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
