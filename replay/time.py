from replay.constants import AnyDataFrame
import pyspark.sql.functions as sf


def get_item_recency(log: AnyDataFrame, decay: float = 30, limit: float = 0.1):
    """
    Calculate item weight showing when the majority of interactions with this item happened.
    :param log: interactions log
    :param decay: number of days after which the weight is reduced by half
    :param limit: minimal value the weight can reach
    :return: DataFrame with item weights
    """
    #  calculate date for each item
    # items =
    return smoothe_time(log, decay, limit)


def smoothe_time(log: AnyDataFrame, decay: float = 30, limit: float = 0.1):
    """
    Weighs `relevance` column with a time-dependent weight.
    :param log: interactions log
    :param decay: number of days after which the weight is reduced by half
    :param limit: minimal value the weight can reach
    :return: modified DataFrame
    """
    log = log.withColumn(
        "timestamp", sf.unix_time(sf.to_timestamp("timestamp"))
    )
    last_date = (
        log.agg({"timestamp": "max"}).collect()[0].asDict()["max(timestamp)"]
    )
    day_in_secs = 86400
    log = log.withColumn(
        "age", (last_date - sf.col("timestamp")) / day_in_secs
    )
    log = log.withColumn("age", 1 - sf.pow("age", 1 / decay))
    log = log.withColumn(
        "age", sf.when(sf.col("age") < limit, limit).otherwise(sf.col("age"))
    )
    log = log.withColumn("relevance", sf.col("relevance") * sf.col("age"))
    return log
