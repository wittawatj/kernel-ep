/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Settings for the multi-class Bayes point machine classifier which affect prediction.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <remarks>
    /// These settings can be modified after training.
    /// </remarks>
    [Serializable]
    public class MulticlassBayesPointMachineClassifierPredictionSettings<TLabel> : BayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// The default number of iterations of the prediction algorithm.
        /// </summary>
        public const int IterationCountDefault = 10;

        /// <summary>
        /// The number of iterations of the prediction algorithm.
        /// </summary>
        private int iterationCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassBayesPointMachineClassifierPredictionSettings{TLabel}"/> class.
        /// </summary>
        internal MulticlassBayesPointMachineClassifierPredictionSettings()
        {
            this.iterationCount = IterationCountDefault;
        }

        /// <summary>
        /// Gets or sets the number of iterations of the prediction algorithm.
        /// </summary>
        public int IterationCount
        {
            get
            {
                return this.iterationCount;
            }

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException("value", "The number of iterations of the prediction algorithm must be positive.");
                }

                this.iterationCount = value;
            }
        }
    }
}