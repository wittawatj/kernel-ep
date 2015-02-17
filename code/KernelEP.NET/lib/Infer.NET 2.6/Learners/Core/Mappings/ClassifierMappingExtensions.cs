/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.Mappings
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Extension methods for the <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> interface.
    /// </summary>
    public static class ClassifierMappingExtensions
    {
        /// <summary>
        /// Safely gets all class labels.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>All possible values of a label.</returns>
        /// <exception cref="MappingException">Thrown if the class labels are null or not unique.</exception>
        public static IEnumerable<TLabel> GetClassLabelsSafe<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues>(
            this IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues> mapping,
            TInstanceSource instanceSource = default(TInstanceSource),
            TLabelSource labelSource = default(TLabelSource))
        {
            IEnumerable<TLabel> classLabels = mapping.GetClassLabels(instanceSource, labelSource);
            if (classLabels == null)
            {
                throw new MappingException("The class labels must not be null.");
            }

            IList<TLabel> classLabelList = classLabels.ToList();
            if (classLabelList.Any(label => label == null))
            {
                throw new MappingException("A class label must not be null.");
            }

            var classLabelSet = new HashSet<TLabel>(classLabelList);
            if (classLabelSet.Count != classLabelList.Count)
            {
                throw new MappingException("All class labels must be unique.");
            }

            return classLabelList;
        }


    }
}
