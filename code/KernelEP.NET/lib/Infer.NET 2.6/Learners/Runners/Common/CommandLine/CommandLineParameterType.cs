/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.Runners
{
    /// <summary>
    /// Represents the type of a command-line parameter.
    /// </summary>
    public enum CommandLineParameterType
    {
        /// <summary>
        /// The parameter is required.
        /// </summary>
        Required,

        /// <summary>
        /// The parameter is optional.
        /// </summary>
        Optional
    }
}