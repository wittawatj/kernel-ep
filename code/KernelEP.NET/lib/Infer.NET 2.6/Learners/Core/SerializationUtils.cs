/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;
    using System.IO;
    using System.Reflection;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Formatters.Binary;

    /// <summary>
    /// Implements various utilities related to learner serialization.
    /// </summary>
    public static class SerializationUtils
    {
        #region Save

        /// <summary>
        /// Persists a learner to a file.
        /// </summary>
        /// <param name="learner">The learner to serialize.</param>
        /// <param name="fileName">The name of the file.</param>
        public static void Save(this ILearner learner, string fileName)
        {
            if (learner == null)
            {
                throw new ArgumentNullException("learner");
            }

            if (fileName == null)
            {
                throw new ArgumentNullException("fileName");
            }

            using (Stream stream = File.Open(fileName, FileMode.Create))
            {
                var formatter = new BinaryFormatter();
                learner.Save(stream, formatter);
            }
        }

        /// <summary>
        /// Serializes a learner to a given stream using a given formatter.
        /// </summary>
        /// <param name="learner">The learner.</param>
        /// <param name="stream">The serialization stream.</param>
        /// <param name="formatter">The formatter.</param>
        public static void Save(this ILearner learner, Stream stream, IFormatter formatter)
        {
            if (learner == null)
            {
                throw new ArgumentNullException("learner");
            }

            if (stream == null)
            {
                throw new ArgumentNullException("stream");
            }

            if (formatter == null)
            {
                throw new ArgumentNullException("formatter");
            }

            AppendVersionMetadata(learner.GetType(), stream, formatter);
            formatter.Serialize(stream, learner);
        }

        #endregion

        #region Load

        /// <summary>
        /// Deserializes a learner from a given stream and formatter.
        /// </summary>
        /// <typeparam name="TLearner">The type of a learner.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized learner object.</returns>
        public static TLearner Load<TLearner>(Stream stream, IFormatter formatter)
        {
            if (stream == null)
            {
                throw new ArgumentNullException("stream");
            }

            if (formatter == null)
            {
                throw new ArgumentNullException("formatter");
            }

            CheckVersion(stream, formatter);
            return (TLearner)formatter.Deserialize(stream);
        }

        /// <summary>
        /// Deserializes a learner from a file.
        /// </summary>
        /// <typeparam name="TLearner">The type of a learner.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized learner object.</returns>
        public static TLearner Load<TLearner>(string fileName)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException("fileName");
            }

            using (Stream stream = File.Open(fileName, FileMode.Open))
            {
                var formatter = new BinaryFormatter();
                return Load<TLearner>(stream, formatter);
            }
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Appends learner metadata required for version checking to a given serialization stream.
        /// </summary>
        /// <param name="type">The type of the learner.</param>
        /// <param name="stream">The serialization stream.</param>
        /// <param name="formatter">The formatter.</param>
        private static void AppendVersionMetadata(Type type, Stream stream, IFormatter formatter)
        {
            formatter.Serialize(stream, type);
            formatter.Serialize(stream, GetSerializationVersion(type));
        }

        /// <summary>
        /// Checks the current version of a learner matches the one in a given serialization stream.
        /// </summary>
        /// <param name="stream">The stream containing the required metadata.</param>
        /// <param name="formatter">The formatter.</param>
        /// <remarks>This method modifies the given stream.</remarks>
        private static void CheckVersion(Stream stream, IFormatter formatter)
        {
            if (formatter == null)
            {
                throw new ArgumentNullException("formatter");
            }

            var type = (Type)formatter.Deserialize(stream);
            var expectedSerializationVersion = GetSerializationVersion(type);

            var actualSerializationVersion = (int)formatter.Deserialize(stream);

            if (expectedSerializationVersion != actualSerializationVersion)
            {
                throw new SerializationException(
                    string.Format(
                        "Serialization version mismatch. Expected: {0}, actual: {1}.",
                        expectedSerializationVersion,
                        actualSerializationVersion));
            }
        }

        /// <summary>
        /// Gets the serialization version of a learner.
        /// </summary>
        /// <param name="memberInfo">The member info of the learner.</param>
        /// <returns>The serialization version of the learner.</returns>
        private static int GetSerializationVersion(MemberInfo memberInfo)
        {
            var serializationVersionAttribute = 
                (SerializationVersionAttribute)Attribute.GetCustomAttribute(memberInfo, typeof(SerializationVersionAttribute));

            if (serializationVersionAttribute == null)
            {
                throw new SerializationException(
                    string.Format(
                        "The {0} must be applied to the learner for serialization and deserialization.",
                        typeof(SerializationVersionAttribute).Name));
            }

            return serializationVersionAttribute.SerializationVersion;
        }

        #endregion
    }
}
