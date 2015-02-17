/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Settings for the Bayes point machine classifier which affect training.
    /// </summary>
    [Serializable]
    public class BayesPointMachineClassifierTrainingSettings
    {
        /// <summary>
        /// The default value indicating whether model evidence is computed during training.
        /// </summary>
        public const bool ComputeModelEvidenceDefault = false;
        
        /// <summary>
        /// The default number of iterations of the training algorithm.
        /// </summary>
        public const int IterationCountDefault = 30;

        /// <summary>
        /// The default number of batches the training data is split into.
        /// </summary>
        public const int BatchCountDefault = 1;

        /// <summary>
        /// Guards the training settings of the Bayes point machine classifier from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Indicates whether model evidence is computed during training.
        /// </summary>
        private bool computeModelEvidence;

        /// <summary>
        /// The number of iterations of the training algorithm.
        /// </summary>
        private int iterationCount;

        /// <summary>
        /// The number of batches the training data is split into.
        /// </summary>
        private int batchCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="BayesPointMachineClassifierTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal BayesPointMachineClassifierTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.computeModelEvidence = ComputeModelEvidenceDefault;
            this.iterationCount = IterationCountDefault;
            this.batchCount = BatchCountDefault;
        }

        /// <summary>
        /// Gets or sets a value indicating whether model evidence is computed during training.
        /// </summary>
        /// <remarks>
        /// This setting cannot be modified after training.
        /// </remarks>
        public bool ComputeModelEvidence
        {
            get
            {
                return this.computeModelEvidence;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                this.computeModelEvidence = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of iterations of the training algorithm.
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
                    throw new ArgumentOutOfRangeException("value", "The number of iterations of the training algorithm must be positive.");
                }

                this.iterationCount = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of batches the training data is split into.
        /// </summary>
        public int BatchCount
        {
            get
            {
                return this.batchCount;
            }

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException("value", "The number of batches must be positive.");
                }

                this.batchCount = value;
            }
        }
    }
}
