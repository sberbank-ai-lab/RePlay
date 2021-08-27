from typing import Any, List, Optional, Set, Union

import numpy as np
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql import Column, DataFrame, Window, functions as sf
from pyspark.sql.types import ArrayType, DoubleType, NumericType
from scipy.sparse import csr_matrix

from replay.constants import NumType, AnyDataFrame
from replay.session_handler import State

# pylint: disable=invalid-name


def convert2spark(data_frame: Optional[AnyDataFrame]) -> Optional[DataFrame]:
    """
    Converts Pandas DataFrame to Spark DataFrame

    :param data_frame: pandas DataFrame
    :return: converted data
    """
    if data_frame is None:
        return None
    if isinstance(data_frame, DataFrame):
        return data_frame
    spark = State().session
    return spark.createDataFrame(data_frame)  # type: ignore


def get_distinct_values_in_column(
    dataframe: DataFrame, column: str
) -> Set[Any]:
    """
    Get unique values from a column as a set.

    :param dataframe: spark DataFrame
    :param column: column name
    :return: set of unique values
    """
    return {
        row[column] for row in (dataframe.select(column).distinct().collect())
    }


def func_get(vector: np.ndarray, i: int) -> float:
    """
    helper function for Spark UDF to get element by index

    :param vector: Scala vector or numpy array
    :param i: index in a vector
    :returns: element value
    """
    return float(vector[i])


def get_top_k_recs(recs: DataFrame, k: int, id_type: str = "id") -> DataFrame:
    """
    Get top k recommendations by `relevance`.

    :param recs: recommendations DataFrame
        `[user_id, item_id, relevance]`
    :param k: length of a recommendation list
    :param id_type: id or idx
    :return: top k recommendations `[user_id, item_id, relevance]`
    """
    window = Window.partitionBy(recs["user_" + id_type]).orderBy(
        recs["relevance"].desc()
    )
    return (
        recs.withColumn("rank", sf.row_number().over(window))
        .filter(sf.col("rank") <= k)
        .drop("rank")
    )


@sf.udf(returnType=DoubleType())  # type: ignore
def vector_dot(one: DenseVector, two: DenseVector) -> float:
    """
    dot product of two column vectors

    >>> from replay.session_handler import State
    >>> from pyspark.ml.linalg import Vectors
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([(Vectors.dense([1.0, 2.0]), Vectors.dense([3.0, 4.0]))])
    ...     .toDF("one", "two")
    ... )
    >>> input_data.dtypes
    [('one', 'vector'), ('two', 'vector')]
    >>> input_data.show()
    +---------+---------+
    |      one|      two|
    +---------+---------+
    |[1.0,2.0]|[3.0,4.0]|
    +---------+---------+
    <BLANKLINE>
    >>> output_data = input_data.select(vector_dot("one", "two").alias("dot"))
    >>> output_data.schema
    StructType(List(StructField(dot,DoubleType,true)))
    >>> output_data.show()
    +----+
    | dot|
    +----+
    |11.0|
    +----+
    <BLANKLINE>

    :param one: vector one
    :param two: vector two
    :returns: dot product
    """
    return float(one.dot(two))


@sf.udf(returnType=VectorUDT())  # type: ignore
def vector_mult(
    one: Union[DenseVector, NumType], two: DenseVector
) -> DenseVector:
    """
    elementwise vector multiplication

    >>> from replay.session_handler import State
    >>> from pyspark.ml.linalg import Vectors
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([(Vectors.dense([1.0, 2.0]), Vectors.dense([3.0, 4.0]))])
    ...     .toDF("one", "two")
    ... )
    >>> input_data.dtypes
    [('one', 'vector'), ('two', 'vector')]
    >>> input_data.show()
    +---------+---------+
    |      one|      two|
    +---------+---------+
    |[1.0,2.0]|[3.0,4.0]|
    +---------+---------+
    <BLANKLINE>
    >>> output_data = input_data.select(vector_mult("one", "two").alias("mult"))
    >>> output_data.schema
    StructType(List(StructField(mult,VectorUDT,true)))
    >>> output_data.show()
    +---------+
    |     mult|
    +---------+
    |[3.0,8.0]|
    +---------+
    <BLANKLINE>

    :param one: vector one
    :param two: vector two
    :returns: result
    """
    return one * two


@sf.udf(returnType=ArrayType(DoubleType()))
def array_mult(first: Column, second: Column):
    """
    elementwise array multiplication

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([([1.0, 2.0], [3.0, 4.0])])
    ...     .toDF("one", "two")
    ... )
    >>> input_data.dtypes
    [('one', 'array<double>'), ('two', 'array<double>')]
    >>> input_data.show()
    +----------+----------+
    |       one|       two|
    +----------+----------+
    |[1.0, 2.0]|[3.0, 4.0]|
    +----------+----------+
    <BLANKLINE>
    >>> output_data = input_data.select(array_mult("one", "two").alias("mult"))
    >>> output_data.schema
    StructType(List(StructField(mult,ArrayType(DoubleType,true),true)))
    >>> output_data.show()
    +----------+
    |      mult|
    +----------+
    |[3.0, 8.0]|
    +----------+
    <BLANKLINE>

    :param first: first array
    :param second: second array
    :returns: result
    """

    return [first[i] * second[i] for i in range(len(first))]


def get_log_info(log: DataFrame) -> str:
    """
    Basic log statistics

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> log = spark.createDataFrame([(1, 2), (3, 4), (5, 2)]).toDF("user_id", "item_id")
    >>> log.show()
    +-------+-------+
    |user_id|item_id|
    +-------+-------+
    |      1|      2|
    |      3|      4|
    |      5|      2|
    +-------+-------+
    <BLANKLINE>
    >>> get_log_info(log)
    'total lines: 3, total users: 3, total items: 2'

    :param log: interaction log containing ``user_id`` and ``item_id``
    :returns: statistics string
    """
    cnt = log.count()
    user_cnt = log.select("user_id").distinct().count()
    item_cnt = log.select("item_id").distinct().count()
    return ", ".join(
        [
            f"total lines: {cnt}",
            f"total users: {user_cnt}",
            f"total items: {item_cnt}",
        ]
    )


def get_stats(
    log: DataFrame, group_by: str = "user_id", target_column: str = "relevance"
) -> DataFrame:
    """
    Calculate log statistics: min, max, mean, median ratings, number of ratings.
    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> test_df = (spark.
    ...   createDataFrame([(1, 2, 1), (1, 3, 3), (1, 1, 2), (2, 3, 2)])
    ...   .toDF("user_id", "item_id", "rel")
    ...   )
    >>> get_stats(test_df, target_column='rel').show()
    +-------+--------+-------+-------+---------+----------+
    |user_id|mean_rel|max_rel|min_rel|count_rel|median_rel|
    +-------+--------+-------+-------+---------+----------+
    |      1|     2.0|      3|      1|        3|         2|
    |      2|     2.0|      2|      2|        1|         2|
    +-------+--------+-------+-------+---------+----------+
    >>> get_stats(test_df, group_by='item_id', target_column='rel').show()
    +-------+--------+-------+-------+---------+----------+
    |item_id|mean_rel|max_rel|min_rel|count_rel|median_rel|
    +-------+--------+-------+-------+---------+----------+
    |      2|     1.0|      1|      1|        1|         1|
    |      3|     2.5|      3|      2|        2|         2|
    |      1|     2.0|      2|      2|        1|         2|
    +-------+--------+-------+-------+---------+----------+

    :param log: spark DataFrame with ``user_id``, ``item_id`` and ``relevance`` columns
    :param group_by: column to group data by, ``user_id`` или ``item_id``
    :param target_column: column with interaction ratings
    :return: spark DataFrame with statistics
    """
    agg_functions = {
        "mean": sf.avg,
        "max": sf.max,
        "min": sf.min,
        "count": sf.count,
    }
    agg_functions_list = [
        func(target_column).alias(str(name + "_" + target_column))
        for name, func in agg_functions.items()
    ]
    agg_functions_list.append(
        sf.expr("percentile_approx({}, 0.5)".format(target_column)).alias(
            "median_" + target_column
        )
    )

    return log.groupBy(group_by).agg(*agg_functions_list)


def check_numeric(feature_table: DataFrame) -> None:
    """
    Check if spark DataFrame columns are of NumericType
    :param feature_table: spark DataFrame
    """
    for column in feature_table.columns:
        if not isinstance(feature_table.schema[column].dataType, NumericType):
            raise ValueError(
                "Column {} has type {}, that is not numeric.".format(
                    column, feature_table.schema[column].dataType
                )
            )


def to_csr(
    log: DataFrame,
    user_count: Optional[int] = None,
    item_count: Optional[int] = None,
) -> csr_matrix:
    """
    Convert DataFrame to csr matrix

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 2], "relevance": [1, 2]})
    >>> data_frame = convert2spark(data_frame)
    >>> m = to_csr(data_frame)
    >>> m.toarray()
    array([[1, 0, 0],
           [0, 0, 2]])

    :param log: interaction log with ``user_idx``, ``item_idx`` and
    ``relevance`` columns
    :param user_count: number of rows in resulting matrix
    :param item_count: number of columns in resulting matrix
    """
    pandas_df = log.select("user_idx", "item_idx", "relevance").toPandas()
    row_count = int(
        user_count
        if user_count is not None
        else pandas_df["user_idx"].max() + 1
    )
    col_count = int(
        item_count
        if item_count is not None
        else pandas_df["item_idx"].max() + 1
    )
    return csr_matrix(
        (
            pandas_df["relevance"],
            (pandas_df["user_idx"], pandas_df["item_idx"]),
        ),
        shape=(row_count, col_count),
    )


def horizontal_explode(
    data_frame: DataFrame,
    column_to_explode: str,
    prefix: str,
    other_columns: List[Column],
) -> DataFrame:
    """
    Transform a column with an array of values into separate columns.
    Each array must contain the same amount of values.

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([(5, [1.0, 2.0]), (6, [3.0, 4.0])])
    ...     .toDF("id_col", "array_col")
    ... )
    >>> input_data.show()
    +------+----------+
    |id_col| array_col|
    +------+----------+
    |     5|[1.0, 2.0]|
    |     6|[3.0, 4.0]|
    +------+----------+
    <BLANKLINE>
    >>> horizontal_explode(input_data, "array_col", "element", [sf.col("id_col")]).show()
    +------+---------+---------+
    |id_col|element_0|element_1|
    +------+---------+---------+
    |     5|      1.0|      2.0|
    |     6|      3.0|      4.0|
    +------+---------+---------+
    <BLANKLINE>

    :param data_frame: input DataFrame
    :param column_to_explode: column with type ``array``
    :param prefix: prefix used for new columns, suffix is an integer
    :param other_columns: columns to select beside newly created
    :returns: DataFrame with elements from ``column_to_explode``
    """
    num_columns = len(data_frame.select(column_to_explode).head()[0])
    return data_frame.select(
        *other_columns,
        *[
            sf.element_at(column_to_explode, i + 1).alias(f"{prefix}_{i}")
            for i in range(num_columns)
        ],
    )


def join_or_return(first, second, on, how):
    """
    Safe wrapper for join of two DataFrames if ``second`` parameter is None it returns ``first``.

    :param first: Spark DataFrame
    :param second: Spark DataFrame
    :param on: name of the join column
    :param how: type of join
    :return: Spark DataFrame
    """
    if second is None:
        return first
    return first.join(second, on=on, how=how)


def fallback(
    base: DataFrame, fill: DataFrame, k: int, id_type: str = "id"
) -> DataFrame:
    """
    Fill missing recommendations for users that have less than ``k`` recomended items.
    Score values for the fallback model may be decreased to preserve sorting.

    :param base: base recommendations that need to be completed
    :param fill: extra recommendations
    :param k: desired recommendation list lengths for each user
    :param id_type: id or idx
    :return: augmented recommendations
    """
    if fill is None:
        return base
    margin = 0.1
    min_in_base = base.agg({"relevance": "min"}).collect()[0][0]
    max_in_fill = fill.agg({"relevance": "max"}).collect()[0][0]
    diff = max_in_fill - min_in_base
    fill = fill.withColumnRenamed("relevance", "relevance_fallback")
    if diff >= 0:
        fill = fill.withColumn(
            "relevance_fallback", sf.col("relevance_fallback") - diff - margin
        )
    recs = base.join(
        fill, on=["user_" + id_type, "item_" + id_type], how="full_outer"
    )
    recs = recs.withColumn(
        "relevance", sf.coalesce("relevance", "relevance_fallback")
    ).select("user_" + id_type, "item_" + id_type, "relevance")
    recs = get_top_k_recs(recs, k, id_type)
    return recs


# pylint: disable=too-many-locals
def get_first_level_model_features(
    model: DataFrame,
    pairs: DataFrame,
    user_features: Optional[DataFrame] = None,
    item_features: Optional[DataFrame] = None,
    add_factors_mult: bool = True,
) -> DataFrame:
    """
    Get user or item features from replay model.
    If a model can return both user and item embeddings, elementwise multiplication can be performed too.
    If a model can't return embedding for specific user/item, zero vector is returned.

    :param model: trained replay model
    :param pairs: user-item pairs to return embeddings for `[user_id/user_idx, item_id/item_idx]`
    :param user_features: user features `[user_id/user_idx, feature_1, ....]`
    :param item_features: item features `[item_id/item_idx, feature_1, ....]`
    :param add_factors_mult: flag to return elementwise multiplication
    :return: Spark DataFrame
    """
    if "user_id" in pairs.columns:
        func_name = "_get_features_wrap"
        suffix = "id"
    else:
        func_name = "_get_features"
        suffix = "idx"

    users = pairs.select("user_{}".format(suffix)).distinct()
    items = pairs.select("item_{}".format(suffix)).distinct()
    user_factors, user_vector_len = getattr(model, func_name)(
        users, user_features
    )
    item_factors, item_vector_len = getattr(model, func_name)(
        items, item_features
    )

    pairs_with_features = join_or_return(
        pairs, user_factors, how="left", on="user_{}".format(suffix)
    )
    pairs_with_features = join_or_return(
        pairs_with_features,
        item_factors,
        how="left",
        on="item_{}".format(suffix),
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
        pairs_with_features.fillna({"user_bias": 0, "item_bias": 0})

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

    for col_name, prefix in factors_to_explode:
        col_set = set(pairs_with_features.columns)
        col_set.remove(col_name)
        pairs_with_features = horizontal_explode(
            data_frame=pairs_with_features,
            column_to_explode=col_name,
            other_columns=[sf.col(column) for column in sorted(list(col_set))],
            prefix=prefix,
        )

    return pairs_with_features
