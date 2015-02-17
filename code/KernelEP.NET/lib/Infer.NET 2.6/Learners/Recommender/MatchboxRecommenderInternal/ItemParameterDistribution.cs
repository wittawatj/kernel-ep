/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;

    using MicrosoftResearch.Infer.Distributions;

    using GaussianArray = Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;

    /// <summary>
    /// Represents the distribution over item parameters.
    /// </summary>
    [Serializable]
    internal class ItemParameterDistribution : EntityParameterDistribution
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ItemParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitDistribution">The distribution over the user traits.</param>
        /// <param name="biasDistribution">The distribution over the user bias.</param>
        public ItemParameterDistribution(GaussianArray traitDistribution, Gaussian biasDistribution)
            : base(traitDistribution, biasDistribution)
        {
        }
    }
}