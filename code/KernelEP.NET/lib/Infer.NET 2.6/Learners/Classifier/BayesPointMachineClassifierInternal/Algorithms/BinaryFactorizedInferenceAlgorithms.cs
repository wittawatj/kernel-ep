/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Runtime.Serialization;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Utils;

    using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<Distributions.Gaussian, double>;

    /// <summary>
    /// An abstract binary Bayes point machine classifier with factorized weight distributions.
    /// <para>
    /// The factorized weight distributions can be interpreted as a <see cref="VectorGaussian"/> distribution 
    /// with diagonal covariance matrix, which ignores possible correlations between weights.
    /// </para>
    /// </summary>
    [Serializable]
    internal abstract class BinaryFactorizedInferenceAlgorithms : InferenceAlgorithms<GaussianArray, bool, Bernoulli>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        protected BinaryFactorizedInferenceAlgorithms(bool computeModelEvidence, bool useSparseFeatures, int featureCount) 
            : base(computeModelEvidence, useSparseFeatures, featureCount)
        {
            // Set the marginal distributions over weights divided by their prior distributions
            // to uniform distributions in the training algorithm (no constraints)
            this.WeightMarginalsDividedByPriors = new GaussianArray(Gaussian.Uniform(), featureCount);

            // Set the constraint distributions over weights to uniform distributions (no constraints)
            this.WeightConstraints = new GaussianArray(Gaussian.Uniform(), featureCount);

            // Initialize the inference algorithms
            this.InitializeInferenceAlgorithms();
        }

        /// <summary>
        /// Gets the distributions over weights as factorized <see cref="Gaussian"/> distributions.
        /// </summary>
        public override IReadOnlyList<IReadOnlyList<Gaussian>> WeightDistributions
        {
            get
            {
                if (this.ReadOnlyWeightMarginals == null)
                {
                    this.ReadOnlyWeightMarginals = new ReadOnlyCollection<IReadOnlyList<Gaussian>>(
                        new IReadOnlyList<Gaussian>[] { new ReadOnlyCollection<Gaussian>(this.WeightMarginals) });
                }

                return this.ReadOnlyWeightMarginals;
            }
        }

        /// <summary>
        /// Sets the labels of the training algorithm to the specified labels.
        /// </summary>
        protected override bool[] Labels
        {
            set
            {
                InferenceAlgorithmUtilities.CheckLabels(value);
                this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.Labels, value);
            }
        }

        /// <summary>
        /// Creates uniform output messages for all training data batches.
        /// </summary>
        /// <param name="batchCount">The number of batches.</param>
        /// <returns>
        /// An array of uniform output messages, one per training data batch, and null if there is only one single batch.
        /// </returns>
        protected override GaussianArray[] CreateUniformBatchOutputMessages(int batchCount)
        {
            return batchCount > 1 ? Util.ArrayInit(batchCount, b => new GaussianArray(Gaussian.Uniform(), this.FeatureCount)) : null;
        }

        /// <summary>
        /// Copies the specified distribution over labels.
        /// </summary>
        /// <param name="labelDistribution">The distribution over labels to be copied.</param>
        /// <returns>The copy of the specified label distribution.</returns>
        protected override Bernoulli CopyLabelDistribution(Bernoulli labelDistribution)
        {
            return new Bernoulli(labelDistribution);
        }

        /// <summary>
        /// Initializes the inference algorithms for training and prediction.
        /// </summary>
        private void InitializeInferenceAlgorithms()
        {
            // Create uniform constraint distributions for the weights of the prediction algorithm (no constraints)
            this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightConstraints, new GaussianArray(Gaussian.Uniform(), this.FeatureCount));
        }

        /// <summary>
        /// Performs all actions required after deserialization.
        /// </summary>
        /// <param name="context">The streaming context.</param>
        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            this.InitializeInferenceAlgorithms();
        }
    }
}
