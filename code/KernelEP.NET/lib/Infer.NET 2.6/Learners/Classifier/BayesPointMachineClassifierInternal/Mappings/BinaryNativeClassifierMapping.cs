/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// The data mapping into the native format of the binary Bayes point machine classifier. 
    /// Chained with the data mapping into the standard format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    internal class BinaryNativeClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>
        : NativeClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, bool>
    {
        /// <summary>
        /// The label of the positive class.
        /// </summary>
        private TLabel positiveClassLabel;

        /// <summary>
        /// The label of the negative class.
        /// </summary>
        private TLabel negativeClassLabel;

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryNativeClassifierMapping{TInstanceSource,TInstance,TLabelSource,TLabel}"/> class.
        /// </summary>
        /// <param name="standardMapping">The mapping for accessing data in standard format.</param>
        public BinaryNativeClassifierMapping(IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> standardMapping)
            : base(standardMapping)
        {
        }

        /// <summary>
        /// Provides the number of classes that the Bayes point machine classifier is used for.
        /// </summary>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>The number of classes that the Bayes point machine classifier is used for.</returns>
        public override int GetClassCount(TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource))
        {
            return 2;
        }

        /// <summary>
        /// Gets the label in standard data format from the specified label in native data format.
        /// </summary>
        /// <param name="nativeLabel">The label in native data format.</param>
        /// <returns>The label in standard data format.</returns>
        public TLabel GetStandardLabel(bool nativeLabel)
        {
            return nativeLabel ? this.positiveClassLabel : this.negativeClassLabel;
        }

        /// <summary>
        /// Sets the class labels.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        public void SetClassLabels(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            // Get the labels of both positive and negative classes
            TLabel[] classLabels = this.StandardMapping.GetClassLabelsSafe(instanceSource, labelSource).ToArray();

            if (classLabels.Length != 2)
            {
                throw new BayesPointMachineClassifierException("There must be precisely two class labels.");
            }

            Array.Sort(classLabels); // Guarantee consistent order of class labels
            this.negativeClassLabel = classLabels[0]; // Guaranteed to be distinct due to safe mapping
            this.positiveClassLabel = classLabels[1]; 
        }

        /// <summary>
        /// Checks that the class labels provided by the mapping are consistent.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>True, if the class labels are consistent and false otherwise.</returns>
        public bool CheckClassLabelConsistency(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            var classLabels = this.StandardMapping.GetClassLabelsSafe(instanceSource, labelSource).ToArray();
            return classLabels.Length == 2 && classLabels[0].Equals(this.negativeClassLabel) && classLabels[1].Equals(this.positiveClassLabel);
        }

        /// <summary>
        /// Gets the labels for the specified instances in native data format.
        /// </summary>
        /// <param name="instances">The instances to get the labels for.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>The labels in native data format.</returns>
        /// <exception cref="BayesPointMachineClassifierException">Thrown if a label is unknown.</exception>
        protected override bool[] GetNativeLabels(
            IList<TInstance> instances, TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            Debug.Assert(instances != null, "The instances must not be null.");
            Debug.Assert(instances.All(instance => instance != null), "An instance must not be null.");
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            // Get all labels and check them
            int labelIndex = 0;
            var nativeLabels = new bool[instances.Count];
            foreach (var instance in instances)
            {
                TLabel standardLabel = this.StandardMapping.GetLabelSafe(instance, instanceSource, labelSource);

                if (standardLabel.Equals(this.positiveClassLabel))
                {
                    nativeLabels[labelIndex] = true;
                }
                else 
                {
                    if (standardLabel.Equals(this.negativeClassLabel))
                    {
                        nativeLabels[labelIndex] = false;
                    }
                    else
                    {
                        throw new BayesPointMachineClassifierException("The class label '" + standardLabel + "' is unknown.");
                    }
                }

                labelIndex++;
            }

            return nativeLabels;
        }
    }
}