/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.Runners
{
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// A command-line module to find related users given a trained recommender model and a dataset.
    /// </summary>
    internal class RecommenderFindRelatedUsersModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string datasetFile = string.Empty;
            string trainedModelFile = string.Empty;
            string predictionsFile = string.Empty;
            int maxRelatedUserCount = 5;
            int minCommonRatingCount = 5;
            int minRelatedUserPoolSize = 5;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--data", "FILE", "Dataset to make predictions for", v => datasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--model", "FILE", "File with trained model", v => trainedModelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "File with generated predictions", v => predictionsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--max-users", "NUM", "Maximum number of related users for a single user; defaults to 5", v => maxRelatedUserCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--min-common-items", "NUM", "Minimum number of items that the query user and the related user should have rated in common; defaults to 5", v => minCommonRatingCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--min-pool-size", "NUM", "Minimum size of the related user pool for a single user; defaults to 5", v => minRelatedUserPoolSize = v, CommandLineParameterType.Optional);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(datasetFile);

            var trainedModel = MatchboxRecommender.Load<RecommenderDataset, User, Item, DummyFeatureSource>(trainedModelFile);
            var evaluator = new RecommenderEvaluator<RecommenderDataset, User, Item, int, int, Discrete>(
                Mappings.StarRatingRecommender.ForEvaluation());
            IDictionary<User, IEnumerable<User>> relatedUsers = evaluator.FindRelatedUsersWhoRatedSameItems(
                trainedModel, testDataset, maxRelatedUserCount, minCommonRatingCount, minRelatedUserPoolSize);
            RecommenderPersistenceUtils.SaveRelatedUsers(predictionsFile, relatedUsers);

            return true;
        }
    }
}
