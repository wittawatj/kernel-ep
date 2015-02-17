/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Learners.Mappings;

    /// <summary>
    /// A binary Bayes point machine classifier with Gaussian prior distributions over weights
    /// which operates on data in the native format of the underlying Infer.NET model.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    [Serializable]
    [SerializationVersion(5)]
    internal class GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource> :
        BinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, GaussianBayesPointMachineClassifierTrainingSettings>
    {
        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="GaussianBinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource}"/> 
        /// class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        public GaussianBinaryNativeDataFormatBayesPointMachineClassifier(IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
            : base(mapping)
        {
            this.Settings = new GaussianBinaryBayesPointMachineClassifierSettings<bool>(() => this.IsTrained);
        }

        /// <summary>
        /// Creates the inference algorithms for the specified feature representation.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        /// <returns>The inference algorithms for the specified feature representation.</returns>
        protected override IInferenceAlgorithms<bool, Bernoulli> CreateInferenceAlgorithms(bool useSparseFeatures, int featureCount)
        {
            return new GaussianBinaryFactorizedInferenceAlgorithms(
                this.Settings.Training.ComputeModelEvidence, useSparseFeatures, featureCount, this.Settings.Training.Advanced.WeightPriorVariance);
        }
    }
}