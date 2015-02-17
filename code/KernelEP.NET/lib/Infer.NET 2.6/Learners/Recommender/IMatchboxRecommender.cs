/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using MicrosoftResearch.Infer.Distributions;

    /// <summary>
    /// Interface to a Matchbox recommender system.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a source of features.</typeparam>
    public interface IMatchboxRecommender<in TInstanceSource, TUser, TItem, in TFeatureSource> :
        IRecommender<TInstanceSource, TUser, TItem, int, Discrete, TFeatureSource>
    {
        /// <summary>
        /// Gets the recommender settings.
        /// </summary>
        new MatchboxRecommenderSettings Settings { get; }
    }
}
