/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Settings of the multi-class Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    public class MulticlassBayesPointMachineClassifierSettings<TLabel> : 
        BayesPointMachineClassifierSettings<TLabel, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassBayesPointMachineClassifierSettings{TLabel}"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the multi-class Bayes point machine classifier is trained.</param>
        internal MulticlassBayesPointMachineClassifierSettings(Func<bool> isTrained)
        {
            this.Training = new BayesPointMachineClassifierTrainingSettings(isTrained);
            this.Prediction = new MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>();
        }
    }
}