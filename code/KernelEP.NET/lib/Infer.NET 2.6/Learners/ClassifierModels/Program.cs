/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;

    /// <summary>
    /// A program which uses Infer.NET to compile various inference algorithms for Bayes point machine classifiers.
    /// </summary>
    internal static class Program
    {
        /// <summary>
        /// The entry point for the program.
        /// </summary>
        /// <param name="args">The array of command-line arguments.</param>
        public static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("Usage: {0} <generated_source_folder>", Environment.GetCommandLineArgs()[0]);
                Environment.Exit(1);
            }

            // The folder to drop the generated inference algorithms to
            string generatedSourceFolder = args[0];

            // Generate all training algorithms
            foreach (bool computeModelEvidence in new[] { false, true })
            {
                foreach (bool useCompoundWeightPriorDistributions in new[] { false, true })
                {
                    foreach (var trainingAlgorithmFactory in AlgorithmFactories.TrainingAlgorithmFactories)
                    {
                        trainingAlgorithmFactory(generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistributions);
                    }
                }                
            }

            // Generate all prediction algorithms
            foreach (var predictionAlgorithmFactory in AlgorithmFactories.PredictionAlgorithmFactories)
            {
                predictionAlgorithmFactory(generatedSourceFolder);
            }
        }
    }
}