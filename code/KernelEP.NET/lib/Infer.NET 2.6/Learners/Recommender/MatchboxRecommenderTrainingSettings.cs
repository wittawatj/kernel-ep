/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// Settings of the Matchbox recommender which affect training.
    /// Cannot be set after training.
    /// </summary>
    [Serializable]
    public class MatchboxRecommenderTrainingSettings
    {
        #region Default values

        /// <summary>
        /// The default value indicating whether user features will be used.
        /// </summary>
        public const bool UseUserFeaturesDefault = false;

        /// <summary>
        /// The default value indicating whether item features will be used.
        /// </summary>
        public const bool UseItemFeaturesDefault = false;

        /// <summary>
        /// The default number of traits.
        /// </summary>
        public const int TraitCountDefault = 4;

        /// <summary>
        /// The default number of inference iterations.
        /// </summary>
        public const int IterationCountDefault = 20;

        /// <summary>
        /// The default number of data batches.
        /// </summary>
        public const int BatchCountDefault = 1;

        #endregion

        #region Fields

        /// <summary>
        /// Guards the training Matchbox recommender settings from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Indicates whether to use explicit user features.
        /// </summary>
        private bool useUserFeatures;

        /// <summary>
        /// Indicates whether to use explicit item features.
        /// </summary>
        private bool useItemFeatures;

        /// <summary>
        /// The number of implicit user or item features (traits) to learn.
        /// </summary>
        private int traitCount;

        /// <summary>
        /// The number of batches the training data is split into.
        /// </summary>
        private int batchCount;

        /// <summary>
        /// The number of inference iterations to run.
        /// </summary>
        private int iterationCount;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        internal MatchboxRecommenderTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.Advanced = new MatchboxRecommenderAdvancedTrainingSettings(isTrained);
            this.useUserFeatures = UseUserFeaturesDefault;
            this.useItemFeatures = UseItemFeaturesDefault;
            this.traitCount = TraitCountDefault;
            this.batchCount = BatchCountDefault;
            this.iterationCount = IterationCountDefault;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the advanced settings of the Matchbox recommender.
        /// </summary>
        public MatchboxRecommenderAdvancedTrainingSettings Advanced { get; private set; }

        /// <summary>
        /// Gets or sets a value indicating whether to use explicit user features.
        /// </summary>
        public bool UseUserFeatures
        { 
            get
            {
                return this.useUserFeatures;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                this.useUserFeatures = value;
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to use explicit item features.
        /// </summary>
        public bool UseItemFeatures
        {
            get
            {
                return this.useItemFeatures;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                this.useItemFeatures = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of implicit user or item features (traits) to learn.
        /// </summary>
        public int TraitCount
        {
            get
            {
                return this.traitCount;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value > 0, "value", "The number of traits must be non-negative.");
                this.traitCount = value;
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
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value > 0, "value", "The number of batches must be positive.");
                this.batchCount = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of inference iterations to run.
        /// </summary>
        public int IterationCount
        {
            get
            {
                return this.iterationCount;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value > 0, "value", "The number of inference iterations must be positive.");
                this.iterationCount = value;
            }
        }

        #endregion
    }
}
