/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.Runners
{
    using System.Collections.Generic;
    using System.IO;

    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// A command-line module to evaluate item recommendation.
    /// </summary>
    internal class RecommenderEvaluateFindRelatedItemsModule : CommandLineModule
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
            int minCommonRatingCount = 5;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--test-data", "FILE", "Test dataset used to obtain ground truth", v => testDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "Predictions to evaluate", v => predictionsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--report", "FILE", "Evaluation report file", v => reportFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--min-common-items", "NUM", "Minimum number of users that the query item and the related item should have been rated by in common; defaults to 5", v => minCommonRatingCount = v, CommandLineParameterType.Optional);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(testDatasetFile);
            IDictionary<Item, IEnumerable<Item>> relatedItems = RecommenderPersistenceUtils.LoadRelatedItems(predictionsFile);

            var evaluatorMapping = Mappings.StarRatingRecommender.ForEvaluation();
            var evaluator = new StarRatingRecommenderEvaluator<RecommenderDataset, User, Item, int>(evaluatorMapping);
            using (var writer = new StreamWriter(reportFile))
            {
                writer.WriteLine(
                    "L1 Sim NDCG: {0:0.000}",
                    evaluator.RelatedItemsMetric(testDataset, relatedItems, minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity));
                writer.WriteLine(
                    "L2 Sim NDCG: {0:0.000}",
                    evaluator.RelatedItemsMetric(testDataset, relatedItems, minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedEuclideanSimilarity));
            }

            return true;
        }
    }
}
