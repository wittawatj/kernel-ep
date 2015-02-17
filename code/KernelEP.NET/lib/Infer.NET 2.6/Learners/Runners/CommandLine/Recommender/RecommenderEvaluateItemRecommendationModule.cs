/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.IO;

    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// A command-line module to evaluate item recommendation.
    /// </summary>
    internal class RecommenderEvaluateItemRecommendationModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string testDatasetFile = string.Empty;
            string predictionsFile = string.Empty;
            string reportFile = string.Empty;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--test-data", "FILE", "Test dataset used to obtain ground truth", v => testDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "Predictions to evaluate", v => predictionsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--report", "FILE", "Evaluation report file", v => reportFile = v, CommandLineParameterType.Required);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(testDatasetFile);
            int minRating = Mappings.StarRatingRecommender.GetRatingInfo(testDataset).MinStarRating;

            IDictionary<User, IEnumerable<Item>> recommendedItems = RecommenderPersistenceUtils.LoadRecommendedItems(predictionsFile);

            var evaluatorMapping = Mappings.StarRatingRecommender.ForEvaluation();
            var evaluator = new StarRatingRecommenderEvaluator<RecommenderDataset, User, Item, int>(evaluatorMapping);
            using (var writer = new StreamWriter(reportFile))
            {
                writer.WriteLine(
                    "NDCG: {0:0.000}", 
                    evaluator.ItemRecommendationMetric(
                    testDataset, 
                    recommendedItems, 
                    Metrics.Ndcg,
                    rating => Convert.ToDouble(rating) - minRating + 1));
            }

            return true;
        }
    }
}
