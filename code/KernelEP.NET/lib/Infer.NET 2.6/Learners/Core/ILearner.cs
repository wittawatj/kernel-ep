/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    /// <summary>
    /// Interface to a learner (something that can do machine learning).
    /// </summary>
    public interface ILearner
    {
        /// <summary>
        /// Gets the capabilities of the learner. These are any properties of the learner that
        /// are not captured by the type signature of the most specific learner interface below.
        /// </summary>
        ICapabilities Capabilities { get; }

        /// <summary>
        /// Gets the settings of the learner. These should be configured once before any 
        /// query methods are called on the learner.
        /// </summary>
        ISettings Settings { get; }
    }
}
