/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;

    using MicrosoftResearch.Infer.Distributions;

    /// <summary>
    /// Settings of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    internal class GaussianBinaryBayesPointMachineClassifierSettings<TLabel> :
        BayesPointMachineClassifierSettings<TLabel, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBinaryBayesPointMachineClassifierSettings{TLabel}"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the binary Bayes point machine classifier is trained.</param>
        internal GaussianBinaryBayesPointMachineClassifierSettings(Func<bool> isTrained)
        {
            this.Training = new GaussianBayesPointMachineClassifierTrainingSettings(isTrained);
            this.Prediction = new BinaryBayesPointMachineClassifierPredictionSettings<TLabel>();
        }
    }
}