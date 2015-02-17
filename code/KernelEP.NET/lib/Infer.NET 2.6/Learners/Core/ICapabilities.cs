/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    /// <summary>
    /// Interface to learner capabilities.
    /// </summary>
    public interface ICapabilities
    {
        /// <summary>
        /// Gets a value indicating whether the learner is precompiled.
        /// </summary>
        /// <remarks>This is currently only relevant for Infer.NET.</remarks>
        bool IsPrecompiled { get; }

        /// <summary>
        /// Gets a value indicating whether the learner supports missing data.
        /// </summary>
        bool SupportsMissingData { get; }

        /// <summary>
        /// Gets a value indicating whether the learner supports sparse data.
        /// </summary>
        bool SupportsSparseData { get; }

        /// <summary>
        /// Gets a value indicating whether the learner supports streamed data.
        /// </summary>
        bool SupportsStreamedData { get; }

        /// <summary>
        /// Gets a value indicating whether the learner supports training on batched data.
        /// </summary>
        bool SupportsBatchedTraining { get; }

        /// <summary>
        /// Gets a value indicating whether the learner supports distributed training.
        /// </summary>
        bool SupportsDistributedTraining { get; }

        /// <summary>
        /// Gets a value indicating whether the learner supports incremental training.
        /// </summary>
        bool SupportsIncrementalTraining { get; }

        /// <summary>
        /// Gets a value indicating whether the learner can compute how well it matches the training data 
        /// (usually for a given set of hyper-parameters).
        /// </summary>
        bool SupportsModelEvidenceComputation { get; }
    }
}