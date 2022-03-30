import logging
from typing import Dict, Union

from pyspark.sql import Window, DataFrame
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame, IntOrList, NumType
from replay.utils import convert2spark
from replay.metrics.base_metric import RecOnlyMetric, process_k


# pylint: disable=too-few-public-methods, arguments-differ, unused-argument
class Coverage(RecOnlyMetric):
    """
    Metric calculation is as follows:

    * take ``K`` recommendations with the biggest ``relevance`` for each ``user_id``
    * count the number of distinct ``item_id`` in these recommendations
    * devide it by the number of items in the whole data set

    """

    def __init__(
        self, log: AnyDataFrame
    ):  # pylint: disable=super-init-not-called
        """
        :param log: pandas or Spark DataFrame
                    It is important for ``log`` to contain all available items.
        """
        self.items = (
            convert2spark(log).select("item_idx").distinct()  # type: ignore
        )
        self.item_count = self.items.count()
        self.logger = logging.getLogger("replay")

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        # not averaged by users
        pass

    # pylint: disable=no-self-use
    def _get_enriched_recommendations(
        self, recommendations: AnyDataFrame, ground_truth: AnyDataFrame
    ) -> DataFrame:
        return convert2spark(recommendations)

    def _conf_interval(
        self,
        recs: AnyDataFrame,
        k_list: IntOrList,
        alpha: float = 0.95,
    ) -> Union[Dict[int, float], float]:
        if isinstance(k_list, int):
            return 0.0
        return {i: 0.0 for i in k_list}

    def _median(
        self,
        recs: AnyDataFrame,
        k_list: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        return self._mean(recs, k_list)

    @process_k
    def _mean(
        self,
        recs: DataFrame,
        k_list: list,
    ) -> Union[Dict[int, NumType], NumType]:
        unknown_item_count = (
            recs.select("item_idx")  # type: ignore
            .distinct()
            .exceptAll(self.items)
            .count()
        )
        if unknown_item_count > 0:
            self.logger.warning(
                "Recommendations contain items that were not present in the log. "
                "The resulting metric value can be more than 1.0 ¯\_(ツ)_/¯"
            )

        best_positions = (
            recs.withColumn(
                "row_num",
                sf.row_number().over(
                    Window.partitionBy("user_idx").orderBy(
                        sf.desc("relevance")
                    )
                ),
            )
            .select("item_idx", "row_num")
            .groupBy("item_idx")
            .agg(sf.min("row_num").alias("best_position"))
            .cache()
        )

        res = {}
        for current_k in k_list:
            res[current_k] = (
                best_positions.filter(
                    sf.col("best_position") <= current_k
                ).count()
                / self.item_count
            )

        best_positions.unpersist()
        return res
