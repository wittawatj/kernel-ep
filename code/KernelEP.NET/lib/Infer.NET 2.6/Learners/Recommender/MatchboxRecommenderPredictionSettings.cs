/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Settings of the Matchbox recommender which affect prediction.
    /// </summary>
    /// <remarks>
    /// These settings can be modified after training.
    /// </remarks>
    [Serializable]
    public class MatchboxRecommenderPredictionSettings
    {
        /// <summary>
        /// The loss function applied when converting an uncertain prediction into a point prediction.
        /// </summary>
        private LossFunction lossFunction;

        /// <summary>
        /// The custom loss function provided by the user.
        /// </summary>
        private Func<int, int, double> customLossFunction;

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderPredictionSettings"/> class.
        /// </summary>
        public MatchboxRecommenderPredictionSettings()
        {
            this.lossFunction = LossFunction.ZeroOne;
            this.customLossFunction = null;
        }

        /// <summary>
        /// Sets the loss function which determines how a prediction in the form of a distribution is converted into a point prediction.
        /// </summary>
        /// <param name="lossFunction">The loss function.</param>
        /// <param name="customLossFunction">
        /// An optional custom loss function. This can only be set when <paramref name="lossFunction"/> is set to 'Custom'. 
        /// The custom loss function returns the loss incurred when choosing an estimate instead of the true value, 
        /// where the first argument is the true value and the second argument is the estimate of the true value.
        /// </param>
        public void SetPredictionLossFunction(LossFunction lossFunction, Func<int, int, double> customLossFunction = null)
        {
            if (lossFunction == LossFunction.Custom)
            {
                if (customLossFunction == null)
                {
                    throw new ArgumentNullException("customLossFunction");
                }
            }
            else
            {
                if (customLossFunction != null)
                {
                    throw new InvalidOperationException("Loss function must be set to '" + LossFunction.Custom + "' when providing a custom loss function.");
                }
            }

            this.customLossFunction = customLossFunction;
            this.lossFunction = lossFunction;
        }

        /// <summary>
        /// Gets the loss function which determines how a prediction in the form of a distribution is converted into a point prediction.
        /// </summary>
        /// <param name="customLossFunction">
        /// The custom loss function on integers. This is <c>null</c> unless the returned <see cref="LossFunction"/> is 'Custom'. 
        /// </param>
        /// <returns>The <see cref="LossFunction"/>.</returns>
        /// <remarks>
        /// A loss function returns the loss incurred when choosing an estimate instead of the true value, 
        /// where the first argument is the true value and the second argument is the estimate of the true value.
        /// </remarks>
        public LossFunction GetPredictionLossFunction(out Func<int, int, double> customLossFunction)
        {
            customLossFunction = this.customLossFunction;
            return this.lossFunction;
        }
    }
}
