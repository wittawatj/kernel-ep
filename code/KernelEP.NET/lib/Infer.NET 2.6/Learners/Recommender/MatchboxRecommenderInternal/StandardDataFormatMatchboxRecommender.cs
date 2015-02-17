/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// Represents a recommender system which operates on data in the standard format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TDataRating">The type of a rating in data.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    [Serializable]
    [SerializationVersion(5)]
    internal class StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource> :
        IMatchboxRecommender<TInstanceSource, TUser, TItem, TFeatureSource>
    {
        #region Fields, constructors, properties

        /// <summary>
        /// The wrapped native Matchbox recommender.
        /// </summary>
        private readonly NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource> recommender;

        /// <summary>
        /// The data mapping to the native Matchbox recommender.
        /// </summary>
        private readonly NativeRecommenderMapping nativeMapping;

        /// <summary>
        /// The mapping to the standard recommender.
        /// </summary>
        private readonly IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, Vector> topLevelMapping;

        /// <summary>
        /// Maps user objects to user identifiers and vice versa.
        /// </summary>
        private readonly IndexedEntitySet<TUser> indexedUserSet;

        /// <summary>
        /// Maps item objects to item identifiers and vice versa.
        /// </summary>
        private readonly IndexedEntitySet<TItem> indexedItemSet;

        /// <summary>
        /// The rating information.
        /// </summary>
        private IStarRatingInfo<TDataRating> starRatingInfo;

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="StandardDataFormatMatchboxRecommender{TInstanceSource,TInstance,TUser,TItem,TDataRating,TFeatureSource}"/> 
        /// class.
        /// </summary>
        /// <param name="topLevelMapping">The mapping used for accessing data.</param>
        internal StandardDataFormatMatchboxRecommender(IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, Vector> topLevelMapping)
        {
            this.topLevelMapping = topLevelMapping;
            this.nativeMapping = new NativeRecommenderMapping(topLevelMapping);
            this.recommender = new NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource>(this.nativeMapping);
            this.indexedUserSet = new IndexedEntitySet<TUser>();
            this.indexedItemSet = new IndexedEntitySet<TItem>();
        }

        #endregion

        #region Explicit ILearner implementation

        /// <summary>
        /// Gets the capabilities of the learner. These are any properties of the learner that
        /// are not captured by the type signature of the most specific learner interface below.
        /// </summary>
        ICapabilities ILearner.Capabilities
        {
            get { return this.recommender.Capabilities; }
        }

        /// <summary>
        /// Gets the settings of the learner. These should be configured once before any 
        /// query methods are called on the learner.
        /// </summary>
        ISettings ILearner.Settings
        {
            get { return this.Settings; }
        }

        #endregion

        #region IMatchboxRecommender implementation

        /// <summary>
        /// Gets the recommender settings.
        /// </summary>
        public MatchboxRecommenderSettings Settings
        {
            get
            {
                return this.recommender.Settings;
            }
        }

        #endregion

        #region IRecommender implementation

        /// <summary>
        /// Gets the capabilities of the recommender.
        /// </summary>
        public IRecommenderCapabilities Capabilities
        {
            get { return this.recommender.Capabilities; }
        }

        /// <summary>
        /// Gets or sets the subset of the users used for related user prediction.
        /// </summary>
        public IEnumerable<TUser> UserSubset
        {
            get
            {
                return this.recommender.UserSubset.Select(u => this.indexedUserSet.Get(u));
            }

            set
            {
                if (value == null)
                {
                    throw new ArgumentNullException("value");
                }

                this.recommender.UserSubset = value.Select(u => this.indexedUserSet.GetId(u));
            }
        }

        /// <summary>
        /// Gets or sets the subset of the items used for related item prediction and recommendation.
        /// </summary>
        public IEnumerable<TItem> ItemSubset
        {
            get
            {
                return this.recommender.ItemSubset.Select(i => this.indexedItemSet.Get(i));
            }

            set
            {
                if (value == null)
                {
                    throw new ArgumentNullException("value");
                }

                this.recommender.ItemSubset = value.Select(i => this.indexedItemSet.GetId(i));
            }
        }

        /// <summary>
        /// Trains the recommender on the given dataset.
        /// </summary>
        /// <param name="instanceSource">The instances of the dataset.</param>
        /// <param name="featureSource">The source of the features for the given instances.</param>
        public void Train(TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            this.starRatingInfo = this.topLevelMapping.GetRatingInfo(instanceSource);
            this.nativeMapping.SetRatingInfo(this.starRatingInfo);
            this.nativeMapping.UseUserFeatures = this.Settings.Training.UseUserFeatures;
            this.nativeMapping.UseItemFeatures = this.Settings.Training.UseItemFeatures;
            this.nativeMapping.SetBatchCount(this.recommender.Settings.Training.BatchCount);

            this.BuildIndexedEntitySets(instanceSource);
            this.nativeMapping.SetIndexedEntitySets(this.indexedUserSet, this.indexedItemSet);

            this.recommender.Train(instanceSource, featureSource);
            this.nativeMapping.SetBatchCount(1);
            this.nativeMapping.SetTrained();
        }

        /// <summary>
        /// Predicts rating for a given user and item.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of the features for the given user and item.</param>
        /// <returns>The predicted rating.</returns>
        public int Predict(TUser user, TItem item, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.recommender.Predict(this.indexedUserSet.GetId(user), this.indexedItemSet.GetId(item), featureSource);
        }

        /// <summary>
        /// Predicts ratings for the instances provided by the given instance source.
        /// </summary>
        /// <param name="instanceSource">The source providing the instances to predict ratings for.</param>
        /// <param name="featureSource">The source of the features for the instances.</param>
        /// <returns>The predicted ratings.</returns>
        public IDictionary<TUser, IDictionary<TItem, int>> Predict(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            // Get result from wrapped implementation
            IDictionary<int, IDictionary<int, int>> result = this.recommender.Predict(instanceSource, featureSource);

            // Map results to TUser/TItem, convert rating back from zero-based representation
            return result.ToDictionary(
                kv => this.indexedUserSet.Get(kv.Key),
                kv => (IDictionary<TItem, int>)kv.Value.ToDictionary(
                    kv2 => this.indexedItemSet.Get(kv2.Key), kv2 => kv2.Value + this.starRatingInfo.MinStarRating));
        }

        /// <summary>
        /// Predicts the distribution of a rating for a given user and item.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of the features for the given user and item.</param>
        /// <returns>The distribution of the rating.</returns>
        public Discrete PredictDistribution(TUser user, TItem item, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.recommender.PredictDistribution(
                this.indexedUserSet.GetId(user), this.indexedItemSet.GetId(item), featureSource);
        }

        /// <summary>
        /// Predicts rating distributions for the instances provided by the given instance source.
        /// </summary>
        /// <param name="instanceSource">The source providing the instances to predict ratings for.</param>
        /// <param name="featureSource">The source of the features for the instances.</param>
        /// <returns>The distributions of the predicted ratings.</returns>
        public IDictionary<TUser, IDictionary<TItem, Discrete>> PredictDistribution(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            // Get result from wrapped implementation
            IDictionary<int, IDictionary<int, Discrete>> result = this.recommender.PredictDistribution(instanceSource, featureSource);

            // Map results to TUser/TItem
            return result.ToDictionary(
                kv => this.indexedUserSet.Get(kv.Key),
                kv => (IDictionary<TItem, Discrete>)kv.Value.ToDictionary(
                    kv2 => this.indexedItemSet.Get(kv2.Key), kv2 => this.RevertRatingDistribution(kv2.Value)));
        }

        /// <summary>
        /// Recommends items to a given user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of the features for the given user and items to recommend.</param>
        /// <returns>The list of recommended items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IEnumerable<TItem> Recommend(
            TUser user, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.recommender.Recommend(this.indexedUserSet.GetId(user), recommendationCount, featureSource)
                .Select(i => this.indexedItemSet.Get(i));
        }

        /// <summary>
        /// Recommends items to a given list of users.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of the features for the users and items.</param>
        /// <returns>The list of recommended items for every user from <paramref name="users"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IDictionary<TUser, IEnumerable<TItem>> Recommend(
            IEnumerable<TUser> users, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            // Obtain the result from the wrapped implementation
            IDictionary<int, IEnumerable<int>> result = this.recommender.Recommend(
                users.Select(u => this.indexedUserSet.GetId(u)), recommendationCount, featureSource);

            // Map the result to TUser/TItem
            return result.ToDictionary(
                kv => this.indexedUserSet.Get(kv.Key),
                kv => kv.Value.Select(i => this.indexedItemSet.Get(i)));
        }

        /// <summary>
        /// Returns a list of users related to <paramref name="user"/>.
        /// </summary>
        /// <param name="user">The user for which related users should be found.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return.</param>
        /// <param name="featureSource">The source of the features for the users.</param>
        /// <returns>The list of related users.</returns>
        /// <remarks>Only users specified in <see cref="UserSubset"/> will be returned.</remarks>
        public IEnumerable<TUser> GetRelatedUsers(
            TUser user, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.recommender.GetRelatedUsers(this.indexedUserSet.GetId(user), relatedUserCount, featureSource)
                .Select(u => this.indexedUserSet.Get(u));
        }

        /// <summary>
        /// Returns a list of related users to each user in <paramref name="users"/>.
        /// </summary>
        /// <param name="users">The list of users for which related users should be found.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return for every user.</param>
        /// <param name="featureSource">The source of the features for the users.</param>
        /// <returns>The list of related users for each user from <paramref name="users"/>.</returns>
        /// <remarks>Only users specified in <see cref="UserSubset"/> will be returned.</remarks>
        public IDictionary<TUser, IEnumerable<TUser>> GetRelatedUsers(
            IEnumerable<TUser> users, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            // Obtain the result from the wrapped implementation
            IDictionary<int, IEnumerable<int>> result = this.recommender.GetRelatedUsers(
                users.Select(u => this.indexedUserSet.GetId(u)), relatedUserCount, featureSource);

            // Map the result to TUser
            return result.ToDictionary(
                kv => this.indexedUserSet.Get(kv.Key),
                kv => kv.Value.Select(u => this.indexedUserSet.Get(u)));
        }

        /// <summary>
        /// Returns a list of items related to <paramref name="item"/>.
        /// </summary>
        /// <param name="item">The item for which related items should be found.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return.</param>
        /// <param name="featureSource">The source of the features for the items.</param>
        /// <returns>The list of related items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> will be returned.</remarks>
        public IEnumerable<TItem> GetRelatedItems(
            TItem item, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.recommender.GetRelatedItems(this.indexedItemSet.GetId(item), relatedItemCount, featureSource)
                .Select(i => this.indexedItemSet.Get(i));
        }

        /// <summary>
        /// Returns a list of related items to each item in <paramref name="items"/>.
        /// </summary>
        /// <param name="items">The list of items for which related items should be found.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return for every item.</param>
        /// <param name="featureSource">The source of the features for the items.</param>
        /// <returns>The list of related items for each item from <paramref name="items"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> will be returned.</remarks>
        public IDictionary<TItem, IEnumerable<TItem>> GetRelatedItems(
            IEnumerable<TItem> items, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            // Obtain the result from the wrapped implementation
            IDictionary<int, IEnumerable<int>> result = this.recommender.GetRelatedItems(
                items.Select(i => this.indexedItemSet.GetId(i)), relatedItemCount, featureSource);

            // Map the result to TItem
            return result.ToDictionary(
                kv => this.indexedItemSet.Get(kv.Key),
                kv => kv.Value.Select(i => this.indexedItemSet.Get(i)));
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Maps a zero-based rating distribution back in the data domain.
        /// </summary>
        /// <param name="zeroBasedRatingDist">The zero-based rating distribution to revert.</param>
        /// <returns>The rating distribution in data domain.</returns>
        private Discrete RevertRatingDistribution(Discrete zeroBasedRatingDist)
        {
            if (this.starRatingInfo.MinStarRating < 0)
            {
                throw new NotSupportedException("Negative ratings are not supported.");
            }

            SparseVector probs = SparseVector.Zero(this.starRatingInfo.MaxStarRating + 1);
            for (int r = this.starRatingInfo.MinStarRating; r <= this.starRatingInfo.MaxStarRating; ++r)
            {
                probs[r] = zeroBasedRatingDist[r - this.starRatingInfo.MinStarRating];
            }

            return new Discrete(probs);
        }

        /// <summary>
        /// Builds the dictionaries which map entity objects to entity identifiers and vice versa.
        /// </summary>
        /// <param name="instanceSource">The source of instances used to build the indexed entity sets.</param>
        private void BuildIndexedEntitySets(TInstanceSource instanceSource)
        {
            IEnumerable<TInstance> instances = this.topLevelMapping.GetInstances(instanceSource);
            foreach (TInstance instance in instances)
            {
                this.indexedUserSet.Add(this.topLevelMapping.GetUser(instanceSource, instance));
                this.indexedItemSet.Add(this.topLevelMapping.GetItem(instanceSource, instance));
            }
        }

        #endregion

        #region NativeRecommenderMapping nested class

        /// <summary>
        /// Represents the mapping of the wrapped Matchbox recommender. Used for chaining with the top level mapping.
        /// </summary>
        [Serializable]
        private class NativeRecommenderMapping : IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource>
        {
            #region Fields, constructors, properties

            /// <summary>
            /// A mapping to the standard data format.
            /// </summary>
            private readonly IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, Vector> topLevelMapping;

            /// <summary>
            /// Protects the updates to the instance representation.
            /// </summary>
            private readonly object instanceRepresentationLock = new object();

            /// <summary>
            /// The last used source of instances.
            /// </summary>
            /// <remarks>Used to avoid redundant conversion.</remarks>
            [NonSerialized]
            private TInstanceSource lastInstanceSource;

            /// <summary>
            /// User identifiers in native data format.
            /// </summary>
            [NonSerialized]
            private int[] userIds;

            /// <summary>
            /// Item identifiers in native data format.
            /// </summary>
            [NonSerialized]
            private int[] itemIds;

            /// <summary>
            /// Ratings in native data format.
            /// </summary>
            [NonSerialized]
            private int[] ratings;

            /// <summary>
            /// The non-zero user feature values in native data format.
            /// </summary>
            [NonSerialized]
            private double[][] nonZeroUserFeatureValues;

            /// <summary>
            /// The non-zero user feature indices in native data format.
            /// </summary>
            [NonSerialized]
            private int[][] nonZeroUserFeatureIndices;

            /// <summary>
            /// The non-zero item feature values in native data format.
            /// </summary>
            [NonSerialized]
            private double[][] nonZeroItemFeatureValues;

            /// <summary>
            /// The non-zero item feature indices in native data format.
            /// </summary>
            [NonSerialized]
            private int[][] nonZeroItemFeatureIndices;

            /// <summary>
            /// The rating info specified during model training.
            /// </summary>
            private IStarRatingInfo<TDataRating> starRatingInfo;

            /// <summary>
            /// The number of batches specified during model training.
            /// </summary>
            private int batchCount;

            /// <summary>
            /// Maps user objects to user identifiers and vice versa.
            /// </summary>
            private IndexedEntitySet<TUser> indexedUserSet;

            /// <summary>
            /// Maps item objects to item identifiers and vice versa.
            /// </summary>
            private IndexedEntitySet<TItem> indexedItemSet;

            /// <summary>
            /// A flag indicating whether this mapping was used for training.
            /// </summary>
            private bool isTrained = false;

            /// <summary>
            /// Indicates whether feature representation was already created.
            /// </summary>
            private bool isFeatureRepresentationCreated = false;

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeRecommenderMapping"/> class.
            /// </summary>
            /// <param name="topLevelMapping">The wrapped mapping for accessing data in standard format.</param>
            public NativeRecommenderMapping(
                IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, Vector> topLevelMapping)
            {
                this.topLevelMapping = topLevelMapping;
            }

            /// <summary>
            /// Gets or sets a value indicating whether to use user features.
            /// </summary>
            public bool UseUserFeatures { private get; set; }

            /// <summary>
            /// Gets or sets a value indicating whether to use item features.
            /// </summary>
            public bool UseItemFeatures { private get; set; }

            #endregion

            #region IMatchboxRecommenderMapping implementation

            /// <summary>
            /// Gets the list of user identifiers from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the user identifiers from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of user identifiers.</returns>
            public IList<int> GetUserIds(TInstanceSource instanceSource, int batchNumber = 0)
            {
                this.UpdateInstanceRepresentation(instanceSource);
                return this.GetBatch(this.userIds, batchNumber);
            }

            /// <summary>
            /// Gets the list of item identifiers from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the item identifiers from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of item identifiers.</returns>
            public IList<int> GetItemIds(TInstanceSource instanceSource, int batchNumber = 0)
            {
                this.UpdateInstanceRepresentation(instanceSource);
                return this.GetBatch(this.itemIds, batchNumber);
            }

            /// <summary>
            /// Gets the list of ratings from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the ratings from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of ratings</returns>
            public IList<int> GetRatings(TInstanceSource instanceSource, int batchNumber = 0)
            {
                Debug.Assert(!this.isTrained, "This method should not be called after training.");
                this.UpdateInstanceRepresentation(instanceSource);
                return this.GetBatch(this.ratings, batchNumber);
            }

            /// <summary>
            /// Gets the number of users from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of users from.</param>
            /// <returns>The number of users.</returns>
            public int GetUserCount(TInstanceSource instanceSource)
            {
                Debug.Assert(!this.isTrained, "This method should not be called after training.");
                return this.indexedUserSet.Count;
            }

            /// <summary>
            /// Gets the number of items from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of items from.</param>
            /// <returns>The number of items.</returns>
            public int GetItemCount(TInstanceSource instanceSource)
            {
                Debug.Assert(!this.isTrained, "This method should not be called after training.");
                return this.indexedItemSet.Count;
            }

            /// <summary>
            /// Gets the number of star ratings.
            /// This is equal to one plus the difference between the maximum and the minimum rating.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of items from.</param>
            /// <returns>The number of ratings.</returns>
            public int GetRatingCount(TInstanceSource instanceSource)
            {
                Debug.Assert(!this.isTrained, "This method should not be called after training.");
                return this.starRatingInfo.MaxStarRating - this.starRatingInfo.MinStarRating + 1;
            }

            /// <summary>
            /// Gets non-zero feature values for all users present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>An array of non-zero user feature arrays where the outer array is indexed by user id.</returns>
            /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
            public IList<IList<double>> GetAllUserNonZeroFeatureValues(TFeatureSource featureSource)
            {
                Debug.Assert(
                    this.UseUserFeatures == true, "This method must not be called when user features are not enabled.");

                this.UpdateFeatureRepresentation(featureSource);
                return this.nonZeroUserFeatureValues;
            }

            /// <summary>
            /// Gets non-zero feature indices for all users present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <returns>An array of non-zero user feature index arrays where the outer array is indexed by user id.</returns>
            /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
            public IList<IList<int>> GetAllUserNonZeroFeatureIndices(TFeatureSource featureSource)
            {
                Debug.Assert(
                    this.UseUserFeatures == true, "This method must not be called when user features are not enabled.");

                this.UpdateFeatureRepresentation(featureSource);
                return this.nonZeroUserFeatureIndices;
            }

            /// <summary>
            /// Gets non-zero feature values for all items present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>An array of non-zero item feature arrays where the outer array is indexed by item id</returns>
            /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
            public IList<IList<double>> GetAllItemNonZeroFeatureValues(TFeatureSource featureSource)
            {
                Debug.Assert(
                    this.UseItemFeatures == true, "This method must not be called when item features are not enabled.");

                this.UpdateFeatureRepresentation(featureSource);
                return this.nonZeroItemFeatureValues;
            }

            /// <summary>
            /// Gets non-zero feature indices for all items present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <returns>An array of non-zero item feature index arrays where the outer array is indexed by item id</returns>
            /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
            public IList<IList<int>> GetAllItemNonZeroFeatureIndices(TFeatureSource featureSource)
            {
                Debug.Assert(
                    this.UseItemFeatures == true, "This method must not be called when item features are not enabled.");

                this.UpdateFeatureRepresentation(featureSource);
                return this.nonZeroItemFeatureIndices;
            }

            /// <summary>
            /// Gets non-zero feature values for a given user.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <param name="userId">The user identifier.</param>
            /// <returns>Non-zero feature values for the user.</returns>
            /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
            public IList<double> GetSingleUserNonZeroFeatureValues(TFeatureSource featureSource, int userId)
            {
                Debug.Assert(
                    this.UseUserFeatures == true, "This method must not be called when user features are not enabled.");

                Vector features = this.topLevelMapping.GetUserFeatures(featureSource, this.indexedUserSet.Get(userId));
                return GetNonZeroValues(features);
            }

            /// <summary>
            /// Gets non-zero feature indices for a given user.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <param name="userId">The user identifier.</param>
            /// <returns>Non-zero feature indices for the user.</returns>
            /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
            public IList<int> GetSingleUserNonZeroFeatureIndices(TFeatureSource featureSource, int userId)
            {
                Debug.Assert(
                    this.UseUserFeatures == true, "This method must not be called when user features are not enabled.");

                Vector features = this.topLevelMapping.GetUserFeatures(featureSource, this.indexedUserSet.Get(userId));
                return GetNonZeroValueIndices(features);
            }

            /// <summary>
            /// Gets non-zero feature values for a given item.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <param name="itemId">The item identifier.</param>
            /// <returns>Non-zero feature values for the item.</returns>
            /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
            public IList<double> GetSingleItemNonZeroFeatureValues(TFeatureSource featureSource, int itemId)
            {
                Debug.Assert(
                    this.UseItemFeatures == true, "This method must not be called when item features are not enabled.");

                Vector features = this.topLevelMapping.GetItemFeatures(featureSource, this.indexedItemSet.Get(itemId));
                return GetNonZeroValues(features);
            }

            /// <summary>
            /// Gets non-zero feature indices for a given item.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <param name="itemId">The item identifier.</param>
            /// <returns>Non-zero feature values for the item.</returns>
            /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
            public IList<int> GetSingleItemNonZeroFeatureIndices(TFeatureSource featureSource, int itemId)
            {
                Debug.Assert(
                    this.UseItemFeatures == true, "This method must not be called when item features are not enabled.");

                Vector features = this.topLevelMapping.GetItemFeatures(featureSource, this.indexedItemSet.Get(itemId));
                return GetNonZeroValueIndices(features);
            }

            #endregion

            #region Helper methods

            /// <summary>
            /// Sets the indexed entity sets.
            /// </summary>
            /// <param name="indexedUserSet">The indexed user set.</param>
            /// <param name="indexedItemSet">The indexed item set.</param>
            public void SetIndexedEntitySets(IndexedEntitySet<TUser> indexedUserSet, IndexedEntitySet<TItem> indexedItemSet)
            {
                this.indexedUserSet = indexedUserSet;
                this.indexedItemSet = indexedItemSet;
            }

            /// <summary>
            /// Sets the rating info.
            /// </summary>
            /// <param name="starRatingInfo">The rating info to set.</param>
            public void SetRatingInfo(IStarRatingInfo<TDataRating> starRatingInfo)
            {
                this.starRatingInfo = starRatingInfo;
            }

            /// <summary>
            /// Sets the number of batches.
            /// </summary>
            /// <param name="batchCount">The number of batches.</param>
            public void SetBatchCount(int batchCount)
            {
                this.batchCount = batchCount;
            }

            /// <summary>
            /// Marks the mapping as one used for training.
            /// </summary>
            public void SetTrained()
            {
                this.isTrained = true;
            }

            /// <summary>
            /// Gets the array of non-zero values in a given vector.
            /// </summary>
            /// <param name="vector">The vector to extract non-zero values from.</param>
            /// <returns>The array of non-zero values.</returns>
            private static double[] GetNonZeroValues(Vector vector)
            {
                return vector.FindAll(x => x != 0.0).Select(vi => vi.Value).ToArray();
            }

            /// <summary>
            /// Gets the array of non-zero value indices in a given vector.
            /// </summary>
            /// <param name="vector">The vector to extract non-zero value indices from.</param>
            /// <returns>The array of non-zero value indices.</returns>
            private static int[] GetNonZeroValueIndices(Vector vector)
            {
                return vector.IndexOfAll(x => x != 0.0).ToArray();
            }

            /// <summary>
            /// Updates the representation of instances in native format.
            /// </summary>
            /// <param name="instanceSource">The source of instances.</param>
            /// <remarks>Call this method every time before accessing instance data.</remarks>
            private void UpdateInstanceRepresentation(TInstanceSource instanceSource)
            {
                lock (this.instanceRepresentationLock)
                {
                    if (ReferenceEquals(this.lastInstanceSource, instanceSource))
                    {
                        return;
                    }

                    Debug.Assert(
                        this.indexedUserSet != null && this.indexedItemSet != null,
                        "Indexed entity sets should be set before updating the instance representation");
                    Debug.Assert(
                        this.starRatingInfo != null,
                        "Rating info should be set before updating the instance representation");

                    this.lastInstanceSource = instanceSource;

                    IEnumerable<TInstance> instances = this.topLevelMapping.GetInstances(instanceSource);
                    var userIdList = new List<int>();
                    var itemIdList = new List<int>();
                    var ratingList = new List<int>();
                    foreach (TInstance instance in instances)
                    {
                        TUser user = this.topLevelMapping.GetUser(instanceSource, instance);
                        userIdList.Add(this.indexedUserSet.GetId(user));

                        TItem item = this.topLevelMapping.GetItem(instanceSource, instance);
                        itemIdList.Add(this.indexedItemSet.GetId(item));

                        // Do not access the ratings during prediction
                        if (!this.isTrained)
                        {
                            TDataRating dataRating = this.topLevelMapping.GetRating(instanceSource, instance);
                            int zeroBasedModelRating = this.starRatingInfo.ToStarRating(dataRating) - this.starRatingInfo.MinStarRating;
                            ratingList.Add(zeroBasedModelRating);
                        }
                    }

                    this.userIds = userIdList.ToArray();
                    this.itemIds = itemIdList.ToArray();
                    this.ratings = ratingList.ToArray();
                }
            }

            /// <summary>
            /// Updates the representation of features in native format.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <remarks>Call this method every time before accessing feature data.</remarks>
            private void UpdateFeatureRepresentation(TFeatureSource featureSource)
            {
                Debug.Assert(this.indexedUserSet != null && this.indexedItemSet != null, "Id mapping should be set before.");
                Debug.Assert(!this.isTrained, "This method is supposed to be called only once during training.");

                if (this.isFeatureRepresentationCreated)
                {
                    return;
                }

                this.isFeatureRepresentationCreated = true;

                if (this.UseUserFeatures)
                {
                    this.nonZeroUserFeatureValues = new double[this.indexedUserSet.Count][];
                    this.nonZeroUserFeatureIndices = new int[this.indexedUserSet.Count][];
                    for (int i = 0; i < this.indexedUserSet.Count; ++i)
                    {
                        Vector features = this.topLevelMapping.GetUserFeatures(featureSource, this.indexedUserSet.Get(i));
                        this.nonZeroUserFeatureValues[i] = GetNonZeroValues(features);
                        this.nonZeroUserFeatureIndices[i] = GetNonZeroValueIndices(features);
                    }
                }

                if (this.UseItemFeatures)
                {
                    this.nonZeroItemFeatureValues = new double[this.indexedItemSet.Count][];
                    this.nonZeroItemFeatureIndices = new int[this.indexedItemSet.Count][];
                    for (int i = 0; i < this.indexedItemSet.Count; ++i)
                    {
                        Vector features = this.topLevelMapping.GetItemFeatures(featureSource, this.indexedItemSet.Get(i));
                        this.nonZeroItemFeatureValues[i] = GetNonZeroValues(features);
                        this.nonZeroItemFeatureIndices[i] = GetNonZeroValueIndices(features);
                    }
                }
            }

            /// <summary>
            /// Gets a particular batch from an array.
            /// </summary>
            /// <typeparam name="T">The type of an array element.</typeparam>
            /// <param name="array">The array.</param>
            /// <param name="batchNumber">The batch number.</param>
            /// <returns>The batch as an array.</returns>
            private IList<T> GetBatch<T>(T[] array, int batchNumber)
            {
                // Avoid the batching performance overhead when the data is not batched.
                if (this.batchCount == 1)
                {
                    Debug.Assert(batchNumber == 0, "If the training data is not batched, the batch number must always be zero.");
                    return array;
                }

                Debug.Assert(
                    batchNumber >= 0 && batchNumber < this.batchCount,
                    "The batch number must be non-negative and less than the total number of batches");

                int batchSize = (int)Math.Ceiling((double)array.Length / this.batchCount);
                int startIndex = batchNumber * batchSize;
                int resultLength = (batchNumber != this.batchCount - 1) ? batchSize : (array.Length - startIndex);

                return new ArraySegment<T>(array, startIndex, resultLength);
            }

            #endregion
        }

        #endregion

        #region IndexedEntitySet nested class

        /// <summary>
        /// Represents a mapping from entity identifiers to entity objects and vice versa.
        /// </summary>
        /// <typeparam name="TEntity">The type of the entity (TUser or TItem).</typeparam>
        [Serializable]
        private class IndexedEntitySet<TEntity>
        {
            /// <summary>
            /// The actual mapping of entity ids to entity objects.
            /// </summary>
            private readonly IndexedSet<TEntity> indexedEntitySet;

            /// <summary>
            /// Initializes a new instance of the <see cref="IndexedEntitySet{TEntity}"/> class.
            /// </summary>
            public IndexedEntitySet()
            {
                this.indexedEntitySet = new IndexedSet<TEntity>();
            }

            /// <summary>
            /// Gets the number of entities in the indexed set.
            /// </summary>
            public int Count
            {
                get
                {
                    return this.indexedEntitySet.Count;
                }
            }

            /// <summary>
            /// Gets an entity object from a given entity identifier.
            /// </summary>
            /// <param name="entityId">The entity identifier.</param>
            /// <returns>The entity.</returns>
            public TEntity Get(int entityId)
            {
                Debug.Assert(entityId < this.Count, "The entity identifier should have already been added.");

                return this.indexedEntitySet.GetElementByIndex(entityId);
            }

            /// <summary>
            /// Adds a new entity to the indexed set.
            /// </summary>
            /// <param name="entity">The entity to add.</param>
            /// <remarks>This method internally checks for containment.</remarks>
            public void Add(TEntity entity)
            {
                if (!this.indexedEntitySet.Contains(entity))
                {
                    this.indexedEntitySet.Add(entity);
                }
            }

            /// <summary>
            /// Gets the id of an entity.
            /// </summary>
            /// <param name="entity">The entity to get an id for.</param>
            /// <returns>The entity id.</returns>
            /// <remarks>This method will add the entity to the indexed set if it is not already there.</remarks>
            public int GetId(TEntity entity)
            {
                int result;

                if (!this.indexedEntitySet.TryGetIndex(entity, out result))
                {
                    result = this.Count;
                    this.Add(entity);
                }

                return result;
            }
        }

        #endregion
    }
}
