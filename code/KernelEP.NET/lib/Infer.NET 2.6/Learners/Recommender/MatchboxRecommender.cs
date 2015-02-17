/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;
    using System.IO;
    using System.Runtime.Serialization;

    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// Matchbox recommender factory.
    /// </summary>
    public static class MatchboxRecommender
    {
        #region Creation
        
        /// <summary>
        /// Creates a Matchbox recommender from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The recommender instance.</returns>
        public static IMatchboxRecommender<TInstanceSource, int, int, TFeatureSource>
            Create<TInstanceSource, TFeatureSource>(
                IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource>(mapping);
        }

        /// <summary>
        /// Creates a recommender from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The recommender instance.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, TFeatureSource>
            Create<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(mapping);
        }

        #endregion

        #region Deserialization
        
        /// <summary>
        /// Deserializes a Matchbox recommender from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, TFeatureSource>
            Load<TInstanceSource, TUser, TItem, TFeatureSource>(string fileName)
        {
            return SerializationUtils.Load<IMatchboxRecommender<TInstanceSource, TUser, TItem, TFeatureSource>>(fileName);
        }
        
        /// <summary>
        /// Deserializes a recommender from a given stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, TFeatureSource>
            Load<TInstanceSource, TUser, TItem, TFeatureSource>(Stream stream, IFormatter formatter)
        {
            return SerializationUtils.Load<IMatchboxRecommender<TInstanceSource, TUser, TItem, TFeatureSource>>(stream, formatter);
        }

        #endregion
    }
}
