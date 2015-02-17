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
    /// Represents distribution over traits and bias of a user or an item.
    /// </summary>
    [Serializable]
    internal abstract class EntityParameterDistribution
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EntityParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitDistribution">The distribution over the entity traits.</param>
        /// <param name="biasDistribution">The distribution over the entity bias.</param>
        protected EntityParameterDistribution(GaussianArray traitDistribution, Gaussian biasDistribution)
        {
            Debug.Assert(traitDistribution != null, "A valid distribution over traits must be provided.");
            
            this.Traits = traitDistribution;
            this.Bias = biasDistribution;
        }

        /// <summary>
        /// Gets the distribution over the traits.
        /// </summary>
        public GaussianArray Traits { get; private set; }

        /// <summary>
        /// Gets the distribution over the bias.
        /// </summary>
        public Gaussian Bias { get; private set; }
    }
}
