/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;

    using MicrosoftResearch.Infer.Distributions;

    using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// A multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over factorized weights.
    /// <para>
    /// The factorized weight distributions can be interpreted as a <see cref="VectorGaussian"/> distribution 
    /// with diagonal covariance matrix, which ignores possible correlations between weights.
    /// </para>
    /// </summary>
    [Serializable]
    internal class GaussianMulticlassFactorizedInferenceAlgorithms : MulticlassFactorizedInferenceAlgorithms
    {
        /// <summary>
        /// The prior distributions over weights for the training algorithm.
        /// </summary>
        private readonly GaussianMatrix weightPriorDistributions;

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianMulticlassFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        /// <param name="classCount">The number of classes that the inference algorithms use.</param>
        /// <param name="weightPriorVariance">The variance of the prior distributions over weights. Defaults to 1.</param>
        public GaussianMulticlassFactorizedInferenceAlgorithms(
            bool computeModelEvidence, bool useSparseFeatures, int featureCount, int classCount, double weightPriorVariance = 1.0)
            : base(computeModelEvidence, useSparseFeatures, featureCount, classCount)
        {
            InferenceAlgorithmUtilities.CheckVariance(weightPriorVariance);

            // Set the prior distributions over weights
            this.weightPriorDistributions = new GaussianMatrix(new GaussianArray(new Gaussian(0.0, weightPriorVariance), featureCount), classCount);
        }

        /// <summary>
        /// Runs the generated training algorithm for the specified features and labels.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <param name="labels">The labels.</param>
        /// <param name="iterationCount">The number of iterations to run the training algorithm for.</param>
        /// <param name="batchNumber">
        /// An optional batch number. Defaults to 0 and is used only if the training data is divided into batches.
        /// </param>
        protected override void TrainInternal(double[][] featureValues, int[][] featureIndexes, int[] labels, int iterationCount, int batchNumber = 0)
        {
            // Update the prior distributions over weights
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightPriors, this.weightPriorDistributions);

            // Run training
            base.TrainInternal(featureValues, featureIndexes, labels, iterationCount, batchNumber);
        }

        #region Inference algorithm generation

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains the Bayes point machine classifier.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the generated training algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        protected override IGeneratedAlgorithm CreateTrainingAlgorithm(bool computeModelEvidence, bool useSparseFeatures)
        {
            return InferenceAlgorithmUtilities.CreateMulticlassTrainingAlgorithm(
                computeModelEvidence,
                useSparseFeatures,
                useCompoundWeightPriorDistributions: false);
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from the Bayes point machine classifier.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the generated prediction algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        protected override IGeneratedAlgorithm CreatePredictionAlgorithm(bool useSparseFeatures)
        {
            return InferenceAlgorithmUtilities.CreateMulticlassPredictionAlgorithm(useSparseFeatures);
        }

        #endregion
    }
}