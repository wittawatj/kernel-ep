/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    using MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// Advanced settings of the Matchbox recommender which affect training.
    /// Cannot be set after training.
    /// </summary>
    [Serializable]
    public class MatchboxRecommenderAdvancedTrainingSettings
    {
        /// <summary>
        /// Guards the training Matchbox recommender settings from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderAdvancedTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        internal MatchboxRecommenderAdvancedTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.User = new UserHyperparameters();
            this.Item = new ItemHyperparameters();
            this.UserFeature = new FeatureHyperparameters();
            this.ItemFeature = new FeatureHyperparameters();
            this.Noise = new NoiseHyperparameters();
        }

        #region User hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the user traits.
        /// </summary>
        public double UserTraitVariance
        {
            get
            {
                return this.User.TraitVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user trait variance must be non-negative.");
                this.User.TraitVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the user bias.
        /// </summary>
        public double UserBiasVariance
        {
            get
            {
                return this.User.BiasVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user bias variance must be non-negative.");
                this.User.BiasVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the prior distribution of the user threshold.
        /// </summary>
        public double UserThresholdPriorVariance
        {
            get
            {
                return this.User.ThresholdPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user threshold prior variance must be non-negative.");
                this.User.ThresholdPriorVariance = value;
            }
        }

        #endregion

        #region Item hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the item traits.
        /// </summary>
        public double ItemTraitVariance
        {
            get
            {
                return this.Item.TraitVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item trait variance must be non-negative.");
                this.Item.TraitVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the item bias.
        /// </summary>
        public double ItemBiasVariance
        {
            get
            {
                return this.Item.BiasVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item bias variance must be non-negative.");
                this.Item.BiasVariance = value;
            }
        }

        #endregion

        #region User feature hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute user traits.
        /// </summary>
        public double UserTraitFeatureWeightPriorVariance
        {
            get
            {
                return this.UserFeature.TraitWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user trait feature weight prior variance must be non-negative.");
                this.UserFeature.TraitWeightPriorVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute user bias.
        /// </summary>
        public double UserBiasFeatureWeightPriorVariance
        {
            get
            {
                return this.UserFeature.BiasWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user bias feature weight prior variance must be non-negative.");
                this.UserFeature.BiasWeightPriorVariance = value;
            }
        }

        #endregion

        #region Item feature hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute item traits.
        /// </summary>
        public double ItemTraitFeatureWeightPriorVariance
        {
            get
            {
                return this.ItemFeature.TraitWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item trait feature weight prior variance must be non-negative.");
                this.ItemFeature.TraitWeightPriorVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute item bias.
        /// </summary>
        public double ItemBiasFeatureWeightPriorVariance
        {
            get
            {
                return this.ItemFeature.BiasWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item bias feature weight prior variance must be non-negative.");
                this.ItemFeature.BiasWeightPriorVariance = value;
            }
        }

        #endregion

        #region Noise hyper-parameters properties

        /// <summary>
        /// Gets or sets the variance of the noise of the user thresholds.
        /// </summary>
        public double UserThresholdNoiseVariance
        {
            get
            {
                return this.Noise.UserThresholdVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The variance of the noise of the user thresholds must be non-negative.");
                this.Noise.UserThresholdVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the affinity noise.
        /// </summary>
        public double AffinityNoiseVariance
        {
            get
            {
                return this.Noise.AffinityVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The variance of the affinity noise must be non-negative.");
                this.Noise.AffinityVariance = value;
            }
        }

        #endregion

        #region Internal properties

        /// <summary>
        /// Gets the user hyper-parameters.
        /// </summary>
        internal UserHyperparameters User { get; private set; }

        /// <summary>
        /// Gets the item hyper-parameters.
        /// </summary>
        internal ItemHyperparameters Item { get; private set; }

        /// <summary>
        /// Gets the user feature hyper-parameters.
        /// </summary>
        internal FeatureHyperparameters UserFeature { get; private set; }

        /// <summary>
        /// Gets the item feature hyper-parameters.
        /// </summary>
        internal FeatureHyperparameters ItemFeature { get; private set; }

        /// <summary>
        /// Gets the noise hyper-parameters.
        /// </summary>
        internal NoiseHyperparameters Noise { get; private set; }

        #endregion
    }
}