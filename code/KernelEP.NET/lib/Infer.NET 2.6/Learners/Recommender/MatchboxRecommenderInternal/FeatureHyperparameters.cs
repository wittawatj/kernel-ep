/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;

    /// <summary>
    /// Represents the feature-related hyper-parameters of the Matchbox recommender.
    /// </summary>
    [Serializable]
    internal class FeatureHyperparameters
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureHyperparameters"/> class.
        /// </summary>
        public FeatureHyperparameters()
        {
            this.TraitWeightPriorVariance = 1.0;
            this.BiasWeightPriorVariance = 1.0;
        }

        /// <summary>
        /// Gets or sets the variance of the weights of the feature contribution to the traits.
        /// </summary>
        public double TraitWeightPriorVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the weights of the feature contribution to the biases.
        /// </summary>
        public double BiasWeightPriorVariance { get; set; }
    }
}
