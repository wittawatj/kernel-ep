/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    /// <summary>
    /// Specifies how metrics are aggregated over the whole dataset.
    /// </summary>
    public enum RecommenderMetricAggregationMethod
    {
        /// <summary>
        /// Metric values for every user-item pair are summed up and then divided by the number of pairs.
        /// </summary>
        Default,

        /// <summary>
        /// Metric values for each user are averaged separately first, then summed up and divided by the number of users.
        /// </summary>
        PerUserFirst
    }
}