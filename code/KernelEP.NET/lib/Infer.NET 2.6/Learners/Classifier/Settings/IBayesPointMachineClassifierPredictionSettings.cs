/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Interface to prediction settings of a Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <remarks>
    /// These settings can be modified after training.
    /// </remarks>
    public interface IBayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// Sets the loss function which determines how a prediction in the form of a distribution is converted into a point prediction.
        /// </summary>
        /// <param name="lossFunction">The loss function.</param>
        /// <param name="customLossFunction">
        /// An optional custom loss function. This can only be set when <paramref name="lossFunction"/> is set to 'Custom'. 
        /// The custom loss function returns the loss incurred when choosing an estimate instead of the true value, 
        /// where the first argument is the true value and the second argument is the estimate of the true value.
        /// </param>
        void SetPredictionLossFunction(LossFunction lossFunction, Func<TLabel, TLabel, double> customLossFunction = null);

        /// <summary>
        /// Gets the loss function which determines how a prediction in the form of a distribution is converted into a point prediction.
        /// </summary>
        /// <param name="customLossFunction">
        /// The custom loss function. This is <c>null</c> unless the returned <see cref="LossFunction"/> is 'Custom'. 
        /// </param>
        /// <returns>The <see cref="LossFunction"/>.</returns>
        /// <remarks>
        /// A loss function returns the loss incurred when choosing an estimate instead of the true value, 
        /// where the first argument is the true value and the second argument is the estimate of the true value.
        /// </remarks>
        LossFunction GetPredictionLossFunction(out Func<TLabel, TLabel, double> customLossFunction);
    }
}