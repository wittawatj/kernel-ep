/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;

    /// <summary>
    /// Represents the item-related hyper-parameters of the Matchbox recommender model.
    /// </summary>
    [Serializable]
    internal class ItemHyperparameters
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ItemHyperparameters"/> class.
        /// </summary>
        public ItemHyperparameters()
        {
            this.TraitVariance = 1.0;
            this.BiasVariance = 1.0;
        }

        /// <summary>
        /// Gets or sets the variance of the item traits.
        /// </summary>
        public double TraitVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the item bias.
        /// </summary>
        public double BiasVariance { get; set; }
    }
}
