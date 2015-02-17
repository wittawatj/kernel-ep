/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Diagnostics;

    using MicrosoftResearch.Infer.Distributions;

    using GaussianArray = Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;

    /// <summary>
    /// Represents the distribution over user parameters.
    /// </summary>
    [Serializable]
    internal class UserParameterDistribution : EntityParameterDistribution
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UserParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitDistribution">The distribution over the user traits.</param>
        /// <param name="biasDistribution">The distribution over the user bias.</param>
        /// <param name="thresholdDistribution">The distribution over the user thresholds.</param>
        public UserParameterDistribution(GaussianArray traitDistribution, Gaussian biasDistribution, GaussianArray thresholdDistribution)
            : base(traitDistribution, biasDistribution)
        {
            Debug.Assert(thresholdDistribution != null, "A valid distribution over thresholds must be provided.");
            
            this.Thresholds = thresholdDistribution;
        }

        /// <summary>
        /// Gets the distribution over thresholds.
        /// </summary>
        public GaussianArray Thresholds { get; private set; }
    }
}