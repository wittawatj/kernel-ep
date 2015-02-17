/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.MatchboxRecommenderInternal
{
    using System;

    /// <summary>
    /// The program.
    /// </summary>
    internal static class Program
    {
        /// <summary>
        /// The entry point for the program.
        /// </summary>
        /// <param name="args">The array of command-line arguments.</param>
        [STAThread]
        public static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("Usage: {0} <generated_source_folder>", Environment.GetCommandLineArgs()[0]);
                Environment.Exit(1);
            }

            string generatedSourceFolder = args[0];
            AlgorithmFactories.GenerateCommunityTrainingAlgorithm(generatedSourceFolder);
            AlgorithmFactories.GenerateRatingPredictionAlgorithm(generatedSourceFolder);
        }
    }
}
