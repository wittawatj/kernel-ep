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
    /// Advanced training settings for a Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
    /// </summary>
    /// <remarks>
    /// These settings cannot be modified after training.
    /// </remarks>
    [Serializable]
    internal class GaussianBayesPointMachineClassifierAdvancedTrainingSettings
    {
        /// <summary>
        /// Guards the training settings of the Bayes point machine classifier from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Gets or sets the variance of the prior distributions over weights of the Bayes point machine classifier.
        /// </summary>
        private double weightPriorVariance;

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBayesPointMachineClassifierAdvancedTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the multi-class Bayes point machine classifier is trained.</param>
        internal GaussianBayesPointMachineClassifierAdvancedTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.weightPriorVariance = 1.0;
        }

        /// <summary>
        /// Gets or sets the variance of the prior distributions over weights of the Bayes point machine classifier.
        /// </summary>
        public double WeightPriorVariance
        {
            get
            {
                return this.weightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();

                if (value < 0)
                {
                    throw new ArgumentOutOfRangeException("value", "The variance of the prior distributions over weights must not be negative.");
                }

                if (double.IsPositiveInfinity(value))
                {
                    throw new ArgumentOutOfRangeException("value", "The variance of the prior distributions over weights must not be infinite.");
                }

                if (double.IsNaN(value))
                {
                    throw new ArgumentOutOfRangeException("value", "The variance of the prior distributions over weights must be a number.");
                }

                this.weightPriorVariance = value;
            }
        }
    }
}