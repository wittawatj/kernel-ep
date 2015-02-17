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
    /// Settings which affect training of the Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
    /// </summary>
    [Serializable]
    internal class GaussianBayesPointMachineClassifierTrainingSettings : BayesPointMachineClassifierTrainingSettings
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBayesPointMachineClassifierTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal GaussianBayesPointMachineClassifierTrainingSettings(Func<bool> isTrained) : base(isTrained)
        {
            this.Advanced = new GaussianBayesPointMachineClassifierAdvancedTrainingSettings(isTrained);
        }

        /// <summary>
        /// Gets the advanced settings of the Bayes point machine classifier.
        /// </summary>
        public GaussianBayesPointMachineClassifierAdvancedTrainingSettings Advanced { get; private set; }
    }
}
