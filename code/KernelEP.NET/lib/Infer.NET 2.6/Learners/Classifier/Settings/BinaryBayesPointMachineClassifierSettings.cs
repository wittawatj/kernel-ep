/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Settings of the binary Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    public class BinaryBayesPointMachineClassifierSettings<TLabel> : 
        BayesPointMachineClassifierSettings<TLabel, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryBayesPointMachineClassifierSettings{TLabel}"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the binary Bayes point machine classifier is trained.</param>
        internal BinaryBayesPointMachineClassifierSettings(Func<bool> isTrained)
        {
            this.Training = new BayesPointMachineClassifierTrainingSettings(isTrained);
            this.Prediction = new BinaryBayesPointMachineClassifierPredictionSettings<TLabel>();
        }
    }
}