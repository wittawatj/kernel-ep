/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;

    /// <summary>
    /// Represents the model noise variance.
    /// </summary>
    [Serializable]
    internal class NoiseHyperparameters
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NoiseHyperparameters"/> class.
        /// </summary>
        public NoiseHyperparameters()
        {
            this.UserThresholdVariance = 0.25;
            this.AffinityVariance = 1.0;
        }

        /// <summary>
        /// Gets or sets the variance of the noise of the user thresholds.
        /// </summary>
        public double UserThresholdVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the affinity noise.
        /// </summary>
        public double AffinityVariance { get; set; }
    }
}
