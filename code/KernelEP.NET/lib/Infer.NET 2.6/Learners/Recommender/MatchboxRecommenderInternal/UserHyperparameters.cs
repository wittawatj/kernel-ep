/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;

    /// <summary>
    /// Represents the user-related hyper-parameters of the Matchbox recommender.
    /// </summary>
    [Serializable]
    internal class UserHyperparameters
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UserHyperparameters"/> class.
        /// </summary>
        public UserHyperparameters()
        {
            this.TraitVariance = 1.0;
            this.BiasVariance = 1.0;
            this.ThresholdPriorVariance = 1.0;
        }

        /// <summary>
        /// Gets or sets the variance of the user traits.
        /// </summary>
        public double TraitVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the user bias.
        /// </summary>
        public double BiasVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the prior distribution of the user threshold.
        /// </summary>
        public double ThresholdPriorVariance { get; set; }
    }
}
