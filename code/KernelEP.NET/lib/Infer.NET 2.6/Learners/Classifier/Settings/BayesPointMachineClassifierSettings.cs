/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Abstract settings of the Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
    [Serializable]
    public abstract class BayesPointMachineClassifierSettings<TLabel, TTrainingSettings, TPredictionSettings> :
        IBayesPointMachineClassifierSettings<TLabel, TTrainingSettings, TPredictionSettings>
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// Gets or sets the settings of the Bayes point machine classifier which affect training.
        /// </summary>
        public TTrainingSettings Training { get; protected set; }

        /// <summary>
        /// Gets or sets the settings of the Bayes point machine classifier which affect prediction.
        /// </summary>
        public TPredictionSettings Prediction { get; protected set; }
    }
}