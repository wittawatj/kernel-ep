/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    /// <summary>
    /// Interface to the settings of an implementation of <see cref="ILearner"/>.
    /// These should be set once to configure the learner before calling any query methods on it.
    /// </summary>
    /// <remarks>
    /// This design is a subject to change.
    /// </remarks>
    public interface ISettings
    {
    }
}