/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Diagnostics;
    using System.Linq;

    using MicrosoftResearch.Infer.Distributions;

    using GaussianArray = Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Represents the distribution over feature weights.
    /// </summary>
    [Serializable]
    internal class FeatureParameterDistribution : ICloneable
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureParameterDistribution"/> class.
        /// </summary>
        public FeatureParameterDistribution()
        {
            this.TraitWeights = new GaussianMatrix(0);
            this.BiasWeights = new GaussianArray(0);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitFeatureWeightDistribution">The distribution over weights of the feature contribution to traits.</param>
        /// <param name="biasFeatureWeightDistribution">The distribution over weights of the feature contribution to biases.</param>
        public FeatureParameterDistribution(GaussianMatrix traitFeatureWeightDistribution, GaussianArray biasFeatureWeightDistribution)
        {
            Debug.Assert(
                (traitFeatureWeightDistribution == null && biasFeatureWeightDistribution == null) ||
                traitFeatureWeightDistribution.All(w => w != null && w.Count == biasFeatureWeightDistribution.Count),
                "The provided distributions should be valid and consistent in the number of features.");
            
            this.TraitWeights = traitFeatureWeightDistribution;
            this.BiasWeights = biasFeatureWeightDistribution;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitCount">The number of traits.</param>
        /// <param name="featureCount">The number of features.</param>
        /// <param name="value">
        /// The value to which each element of the 
        /// contained distributions will be initialized.
        /// </param>
        public FeatureParameterDistribution(int traitCount, int featureCount, Gaussian value)
        {
            this.TraitWeights = new GaussianMatrix(new GaussianArray(value, featureCount), traitCount);
            this.BiasWeights = new GaussianArray(value, featureCount);
        }

        /// <summary>
        /// Gets the number of features.
        /// </summary>
        public int FeatureCount
        {
            get { return this.BiasWeights.Count; }
        }

        /// <summary>
        /// Gets the distribution over weights of the feature contribution to traits.
        /// </summary>
        public GaussianMatrix TraitWeights { get; private set; }

        /// <summary>
        /// Gets the distribution over weights of the feature contribution to biases.
        /// </summary>
        public GaussianArray BiasWeights { get; private set; }

        /// <summary>
        /// Creates a new object that is a copy of the current instance.
        /// </summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        public object Clone()
        {
            return new FeatureParameterDistribution
            {
                TraitWeights = (GaussianMatrix)this.TraitWeights.Clone(),
                BiasWeights = (GaussianArray)this.BiasWeights.Clone(),
            };
        }
    }
}