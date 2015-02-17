namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    internal static class ExperimentalFactor
    {
        [ParameterNames("items", "array", "indices", "dict")]
        public static T[] GetItemsWithDictionary<T>(IList<T> array, IList<string> indices, IDictionary<string, int> dict)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                result[i] = array[dict[indices[i]]];
            }
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="ExperimentalFactor.GetItemsWithDictionary{T}(IList{T}, IList{String}, IDictionary{String, int})" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(ExperimentalFactor), "GetItemsWithDictionary<>")]
    [Quality(QualityBand.Experimental)]
    [Buffers("marginal")]
    internal static class GetItemsWithDictionaryOp<T>
    {
        /// <summary>Initialize the buffer <c>marginal</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Initial value of buffer <c>marginal</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

        /// <summary>Update the buffer <c>marginal</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="dict">Constant value for <c>dict</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        [SkipIfAllUniform]
        public static ArrayType Marginal<ArrayType, DistributionType>(
            ArrayType array, [NoInit] IList<DistributionType> items, IList<string> indices, IDictionary<string, int> dict, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            result.SetTo(array);
            for (int i = 0; i < indices.Count; i++)
            {
                int index = dict[indices[i]];
                DistributionType value = result[index];
                value.SetToProduct(value, items[i]);
                result[index] = value;
            }
            return result;
        }

        /// <summary />
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <param name="to_item" />
        /// <param name="item" />
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="dict">Constant value for <c>dict</c>.</param>
        /// <param name="resultIndex">Index of the <c>marginal</c> for which a message is desired.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, DistributionType>(
            ArrayType result, DistributionType to_item, [SkipIfUniform] DistributionType item, IList<string> indices, IDictionary<string, int> dict, int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            int i = resultIndex;
            int index = dict[indices[i]];
            DistributionType value = result[index];
            value.SetToProduct(to_item, item);
            result[index] = value;
            return result;
        }

        /// <summary>EP message to <c>items</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="marginal">Buffer <c>marginal</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="dict">Constant value for <c>dict</c>.</param>
        /// <param name="resultIndex">Index of the <c>items</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>items</c> as the random arguments are varied. The formula is <c>proj[p(items) sum_(array) p(array) factor(items,array,indices,dict)]/p(items)</c>.</para>
        /// </remarks>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        public static DistributionType ItemsAverageConditional<ArrayType, DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [IgnoreDependency] ArrayType array, // must have an (unused) 'array' argument to determine the type of 'marginal' buffer
            [SkipIfAllUniform] ArrayType marginal,
            IList<string> indices,
            IDictionary<string, int> dict,
            int resultIndex,
            DistributionType result)
            where ArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            int index = dict[indices[i]];
            result.SetToRatio(marginal[index], items);
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="dict">Constant value for <c>dict</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(items) p(items) factor(items,array,indices,dict)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="items" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<DistributionType> items, IList<string> indices, IDictionary<string, int> dict, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                int index = dict[indices[i]];
                DistributionType value = result[index];
                value.SetToProduct(value, items[i]);
                result[index] = value;
            }
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="dict">Constant value for <c>dict</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(items) p(items) factor(items,array,indices,dict)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="items" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<T> items, IList<string> indices, IDictionary<string, int> dict, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                int index = dict[indices[i]];
                DistributionType value = result[index];
                value.Point = items[i];
                result[index] = value;
            }
            return result;
        }
    }
}
