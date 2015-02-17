/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Sets the serialization version of the learner.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Interface, Inherited = true, AllowMultiple = false)]
    public class SerializationVersionAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SerializationVersionAttribute"/> class.
        /// </summary>
        /// <param name="serializationVersion">The serialization version of the learner.</param>
        public SerializationVersionAttribute(int serializationVersion)
        {
            this.SerializationVersion = serializationVersion;
        }

        /// <summary>
        /// Gets the serialization version of the learner.
        /// </summary>
        public int SerializationVersion { get; private set; }
    }
}
