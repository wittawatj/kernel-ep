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
    /// A command-line module to predict ratings given a trained recommender model and a dataset.
    /// </summary>
    internal class RecommenderRecommendItemsModule : CommandLineModule
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
            int maxRecommendedItemCount = 5;
            int minRecommendationPoolSize = 5;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--data", "FILE", "Dataset to make predictions for", v => datasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--model", "FILE", "File with trained model", v => trainedModelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "File with generated predictions", v => predictionsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--max-items", "NUM", "Maximum number of items to recommend; defaults to 5", v => maxRecommendedItemCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--min-pool-size", "NUM", "Minimum size of the recommendation pool for a single user; defaults to 5", v => minRecommendationPoolSize = v, CommandLineParameterType.Optional);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(datasetFile);

            var trainedModel = MatchboxRecommender.Load<RecommenderDataset, User, Item, DummyFeatureSource>(trainedModelFile);
            var evaluator = new RecommenderEvaluator<RecommenderDataset, User, Item, int, int, Discrete>(
                Mappings.StarRatingRecommender.ForEvaluation());
            IDictionary<User, IEnumerable<Item>> itemRecommendations = evaluator.RecommendRatedItems(
                trainedModel, testDataset, maxRecommendedItemCount, minRecommendationPoolSize);
            RecommenderPersistenceUtils.SaveRecommendedItems(predictionsFile, itemRecommendations);

            return true;
        }
    }
}
