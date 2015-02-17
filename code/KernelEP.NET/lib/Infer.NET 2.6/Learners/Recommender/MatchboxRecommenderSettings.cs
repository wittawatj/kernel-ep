/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Settings of the Matchbox recommender (settable by the developer).
    /// </summary>
    [Serializable]
    public class MatchboxRecommenderSettings : ISettings
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderSettings"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        public MatchboxRecommenderSettings(Func<bool> isTrained)
        {
            this.Training = new MatchboxRecommenderTrainingSettings(isTrained);
            this.Prediction = new MatchboxRecommenderPredictionSettings();
        }

        /// <summary>
        /// Gets the settings of the Matchbox recommender which affect training.
        /// </summary>
        public MatchboxRecommenderTrainingSettings Training { get; private set; }

        /// <summary>
        /// Gets the settings of the Matchbox recommender which affect prediction.
        /// </summary>
        public MatchboxRecommenderPredictionSettings Prediction { get; private set; }
    }
}