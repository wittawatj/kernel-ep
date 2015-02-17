/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Runtime.Serialization;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal;
    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// The Bayes point machine classifier factory.
    /// </summary>
    public static class BayesPointMachineClassifier
    {
        #region Public factory methods

        /// <summary>
        /// Creates a binary Bayes point machine classifier from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            CreateBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new CompoundBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            CreateMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new CompoundMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a binary Bayes point machine classifier from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new CompoundBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new CompoundMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        #endregion  

        #region Deserialization

        /// <summary>
        /// Deserializes a Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>
            Load<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>(string fileName)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
            where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>>(fileName);
        }

        /// <summary>
        /// Deserializes a Bayes point machine classifier from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>
            Load<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>(Stream stream, IFormatter formatter)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
            where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(string fileName)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(Stream stream, IFormatter formatter)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(string fileName)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a stream and a formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(Stream stream, IFormatter formatter)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a stream and format.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return SerializationUtils.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        #endregion

        #region Internal factory methods

        /// <summary>
        /// Creates a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            CreateGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            CreateGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new GaussianMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new GaussianBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        internal static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException("mapping");
            }

            return new GaussianMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        #endregion
    }
}
