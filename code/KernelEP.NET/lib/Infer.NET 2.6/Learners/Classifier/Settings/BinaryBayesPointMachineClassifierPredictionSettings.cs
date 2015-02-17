/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Settings for the binary Bayes point machine classifier which affect prediction.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <remarks>
    /// These settings can be modified after training.
    /// </remarks>
    [Serializable]
    public class BinaryBayesPointMachineClassifierPredictionSettings<TLabel> : BayesPointMachineClassifierPredictionSettings<TLabel>
    {
    }
}