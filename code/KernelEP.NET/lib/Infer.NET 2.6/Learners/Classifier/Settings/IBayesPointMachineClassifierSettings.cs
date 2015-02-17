/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    /// <summary>
    /// Interface to settings of a Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
    public interface IBayesPointMachineClassifierSettings<TLabel, out TTrainingSettings, out TPredictionSettings> : ISettings
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// Gets the settings of the Bayes point machine classifier which affect training.
        /// </summary>
        TTrainingSettings Training { get; }

        /// <summary>
        /// Gets the settings of the Bayes point machine classifier which affect prediction.
        /// </summary>
        TPredictionSettings Prediction { get; }
    }
}