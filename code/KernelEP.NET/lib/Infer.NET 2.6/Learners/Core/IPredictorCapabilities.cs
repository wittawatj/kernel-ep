/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    /// <summary>
    /// Interface to predictor capabilities.
    /// </summary>
    public interface IPredictorCapabilities : ICapabilities
    {
        /// <summary>
        /// Gets a value indicating whether the predictor can compute predictive point estimates from a user-defined loss function.
        /// </summary>
        bool SupportsCustomPredictionLossFunction { get; }
    }
}
