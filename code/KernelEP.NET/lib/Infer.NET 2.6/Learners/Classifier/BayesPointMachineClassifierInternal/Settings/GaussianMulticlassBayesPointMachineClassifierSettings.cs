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
    /// Settings of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    internal class GaussianMulticlassBayesPointMachineClassifierSettings<TLabel> :
        BayesPointMachineClassifierSettings<TLabel, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianMulticlassBayesPointMachineClassifierSettings{TLabel}"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the multi-class Bayes point machine classifier is trained.</param>
        internal GaussianMulticlassBayesPointMachineClassifierSettings(Func<bool> isTrained)
        {
            this.Training = new GaussianBayesPointMachineClassifierTrainingSettings(isTrained);
            this.Prediction = new MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>();
        }
    }
}