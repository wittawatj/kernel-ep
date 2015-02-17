/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Learners.Mappings;

    /// <summary>
    /// An abstract base class for a binary Bayes point machine classifier which operates on 
    /// data in the native format of the underlying Infer.NET model.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    [Serializable]
    internal abstract class BinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TTrainingSettings> :
        NativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
    {
        #region Fields, constructor, properties

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="BinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TTrainingSettings}"/> 
        /// class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        protected BinaryNativeDataFormatBayesPointMachineClassifier(IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
            : base(mapping)
        {
        }

        /// <summary>
        /// Gets the point estimator for the label distribution.
        /// </summary>
        protected override Func<Bernoulli, bool> PointEstimator
        {
            get
            {
                Func<bool, bool, double> customLossFunction;
                LossFunction lossFunction = this.Settings.Prediction.GetPredictionLossFunction(out customLossFunction);
                return (lossFunction == LossFunction.Custom) ? 
                    Learners.PointEstimator.ForBernoulli(customLossFunction) : Learners.PointEstimator.ForBernoulli(lossFunction);
            }
        }

        #endregion

        #region Template methods for inference algorithms

        /// <summary>
        /// Runs the prediction algorithm on the specified data.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <returns>The predictive distribution over labels.</returns>
        protected override IEnumerable<Bernoulli> PredictDistribution(double[][] featureValues, int[][] featureIndexes)
        {
            return this.InferenceAlgorithms.PredictDistribution(featureValues, featureIndexes, 1);
        }

        #endregion
    }
}