// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.GetItems{T}(IList{T}, IList{int})" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Factor), "GetItems<>", Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("marginal")]
    public static class GetItemsOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(IList<T> items, IList<T> array, IList<int> indices)
        {
            IEqualityComparer<T> equalityComparer = Utils.Util.GetEqualityComparer<T>();
            for (int i = 0; i < items.Count; i++)
            {
                if (!equalityComparer.Equals(items[i], array[indices[i]])) return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(IList<T> items, IList<T> array, IList<int> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(IList<T> items, IList<T> array, IList<int> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<DistributionType> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value;
                if (!productBefore.TryGetValue(indices[i], out value))
                {
                    value = (DistributionType) array[indices[i]].Clone();
                }
                z += value.GetLogAverageOf(items[i]);
                value.SetToProduct(value, items[i]);
                productBefore[indices[i]] = value;
            }
            return z;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IList<DistributionType> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<T> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value;
                if (!productBefore.TryGetValue(indices[i], out value))
                {
                    value = array[indices[i]];
                }
                z += value.GetLogProb(items[i]);
                value.Point = items[i];
                productBefore[indices[i]] = value;
            }
            return z;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(IList<T> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return LogAverageFactor<DistributionType>(items, array, indices);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IList<T> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<DistributionType> items, IList<T> array, IList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
                z += items[i].GetLogProb(array[indices[i]]);
            return z;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<DistributionType>(IList<DistributionType> items, IList<T> array, IList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            return 0.0;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IList<DistributionType> items, IList<T> array, IList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            return 0.0;
        }


#if true
        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="to_items">Previous outgoing message to <c>items</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(
            IList<DistributionType> items, IList<DistributionType> array, IList<int> indices, IList<DistributionType> to_items)
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
        {
            // this code is adapted from GetItemsOp
            double z = 0.0;
            if (items.Count <= 1) return 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value;
                if (!productBefore.TryGetValue(indices[i], out value))
                {
                    value = (DistributionType) array[indices[i]].Clone();
                }
                z += value.GetLogAverageOf(items[i]);
                value.SetToProduct(value, items[i]);
                productBefore[indices[i]] = value;
                z -= to_items[i].GetLogAverageOf(items[i]);
            }
            return z;
        }
#else
    /// <summary>
    /// Evidence message for EP
    /// </summary>
    /// <param name="items">Incoming message from 'items'.</param>
    /// <param name="array">Incoming message from 'array'.</param>
    /// <param name="indices">Constant value for 'indices'.</param>
    /// <returns>Logarithm of the factor's contribution the EP model evidence</returns>
    /// <remarks><para>
    /// The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>.
    /// Adding up these values across all factors and variables gives the log-evidence estimate for EP.
    /// </para></remarks>
        public static double LogEvidenceRatio<DistributionType, ArrayType>(ArrayType items, IList<DistributionType> array, IList<int> indices)
            where ArrayType : IList<DistributionType>, ICloneable
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
        {
            // result is LogAverageFactor - sum_i toItems[i].LogAverageOf(items[i])
            // we compute this efficiently by constructing toItems[i] by dynamic programming.
            // toItems[i] = productBefore[i] * productAfter[i].
            double z = 0.0;
            if (items.Count <= 1) return 0.0;
            DistributionType uniform = (DistributionType)items[0].Clone();
            uniform.SetToUniform();
            ArrayType productBefore = (ArrayType)items.Clone();
            ArrayType productAfter = (ArrayType)items.Clone();
            Dictionary<int,int> indexToItem = new Dictionary<int, int>();
            for (int i = 0; i < indices.Count; i++) {
                int previousItem;
                if (!indexToItem.TryGetValue(indices[i], out previousItem)) {
                    // no previous item with this index
                    productBefore[i] = array[indices[i]];
                } else {
                    DistributionType temp = productBefore[i];
                    temp.SetToProduct(productBefore[previousItem], items[previousItem]);
                    productBefore[i] = temp;
                }
                z += productBefore[i].GetLogAverageOf(items[i]);
                indexToItem[indices[i]] = i;
            }
            indexToItem.Clear();
            for (int i = indices.Count - 1; i >= 0; i--) {
                int itemAfter;
                if (!indexToItem.TryGetValue(indices[i], out itemAfter)) {
                    // no item after with this index
                    productAfter[i] = uniform;
                } else {
                    DistributionType temp = productAfter[i];
                    temp.SetToProduct(productAfter[itemAfter], items[itemAfter]);
                    productAfter[i] = temp;
                }
                DistributionType toItem = (DistributionType)items[i].Clone();
                toItem.SetToProduct(productBefore[i], productAfter[i]);
                z -= toItem.GetLogAverageOf(items[i]);
                indexToItem[indices[i]] = i;
            }
            return z;
        }
#endif

        /// <summary>Initialize the buffer <c>marginal</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Initial value of buffer <c>marginal</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

        /// <summary>Update the buffer <c>marginal</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [SkipIfAllUniform]
        public static ArrayType Marginal<ArrayType, DistributionType>(
            ArrayType array, [NoInit] IList<DistributionType> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetTo(array);
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        /// <summary />
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <param name="to_item" />
        /// <param name="item" />
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="resultIndex">Index of the <c>marginal</c> for which a message is desired.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, DistributionType>(
            ArrayType result, DistributionType to_item, [SkipIfUniform] DistributionType item, IList<int> indices, int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            int i = resultIndex;
            DistributionType value = result[indices[i]];
            value.SetToProduct(to_item, item);
            result[indices[i]] = value;
            return result;
        }

        /// <summary>EP message to <c>items</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="marginal">Buffer <c>marginal</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="resultIndex">Index of the <c>items</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>items</c> as the random arguments are varied. The formula is <c>proj[p(items) sum_(array) p(array) factor(items,array,indices)]/p(items)</c>.</para>
        /// </remarks>
        /// <typeparam name="ArrayType">The type of an array for the marginal and the result.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<ArrayType, DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [IgnoreDependency] ArrayType array, // must have an (unused) 'array' argument to determine the type of 'marginal' buffer
            [SkipIfAllUniform] ArrayType marginal,
            IList<int> indices,
            int resultIndex, 
            DistributionType result)
            where ArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToRatio(marginal[indices[i]], items);
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(items) p(items) factor(items,array,indices)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="items" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<DistributionType> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(items) p(items) factor(items,array,indices)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="items" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<T> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.Point = items[i];
                result[indices[i]] = value;
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <summary>VMP message to <c>items</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="resultIndex">Index of the <c>items</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>items</c> as the random arguments are varied. The formula is <c>proj[sum_(array) p(array) factor(items,array,indices)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageLogarithm<DistributionType>(
            [SkipIfAllUniform] IList<DistributionType> array, IList<int> indices, int resultIndex, DistributionType result)
            where DistributionType : SettableTo<DistributionType>
        {
            int i = resultIndex;
            result.SetTo(array[indices[i]]);
            return result;
        }

        public static ResultType ItemsAverageLogarithm2<DistributionType, ResultType>(
            [SkipIfAllUniform] IList<DistributionType> array, IList<int> indices, ResultType result)
            where ResultType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[i];
                value.SetTo(array[indices[i]]);
                result[i] = value;
            }
            return result;
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageLogarithmInit<TDist>(
            [IgnoreDependency] DistributionStructArray<TDist, T> array, IList<int> indices)
            where TDist : struct,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageLogarithmInit<TDist>(
            [IgnoreDependency] DistributionRefArray<TDist, T> array, IList<int> indices)
            where TDist : class,
                SettableTo<TDist>,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>items</c> integrated out. The formula is <c>sum_items p(items) factor(items,array,indices)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="items" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<DistributionType> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>items</c> integrated out. The formula is <c>sum_items p(items) factor(items,array,indices)</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(IList<T> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.GetItems{T}(IList{T}, IList{int})" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of a list element.</typeparam>
    [FactorMethod(typeof(Factor), "GetItems<>", Default = false)]
    [Buffers("partial")]
    [Quality(QualityBand.Mature)]
    public static class GetItemsOp2<T>
    {
        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageConditionalInit<TDist>(
            [IgnoreDependency] DistributionStructArray<TDist, T> array, IList<int> indices)
            where TDist : struct,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageConditionalInit<TDist>(
            [IgnoreDependency] DistributionRefArray<TDist, T> array, IList<int> indices)
            where TDist : class,
                SettableTo<TDist>,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageConditional2<DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [SkipIfAllUniform] IList<DistributionType> array,
            [Fresh] IList<DistributionType> to_array,
            IList<int> indices,
            int resultIndex,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToProduct(to_array[indices[i]], array[indices[i]]);
            result.SetToRatio(result, items);
            return result;
        }

        /// <summary>EP message to <c>items</c>.</summary>
        /// <param name="partial">Buffer <c>partial</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="resultIndex">Index of the <c>items</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>items</c> as the random arguments are varied. The formula is <c>proj[p(items) sum_(array) p(array) factor(items,array,indices)]/p(items)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<DistributionType>(
            [Indexed] DistributionType partial,
            [SkipIfAllUniform] IList<DistributionType> array,
            IList<int> indices,
            int resultIndex,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToProduct(partial, array[indices[i]]);
            return result;
        }

        /// <summary />
        /// <param name="partial">Buffer <c>partial</c>.</param>
        /// <param name="item" />
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ArrayIncrement<DistributionType>(DistributionType partial,
            [SkipIfUniform] DistributionType item,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>
        {
            result.SetToProduct(partial, item);
            return result;
        }

        /// <summary>Initialize the buffer <c>partial</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <returns>Initial value of buffer <c>partial</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="ArrayType">The type of a distribution array.</typeparam>
        [Skip]
        public static ArrayType PartialInit<ArrayType>(ArrayType items)
            where ArrayType : ICloneable, SettableToUniform
        {
            ArrayType result = (ArrayType)items.Clone();
            result.SetToUniform();
            return result;
        }

        /// <summary>Update the buffer <c>partial</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="to_array">Outgoing message to <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="resultIndex">Index of the <c>partial</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType Partial<DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [Fresh] IList<DistributionType> to_array,
            IList<int> indices,
            int resultIndex,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToRatio(to_array[indices[i]], items);
            return result;
        }    
    }

    //[FactorMethod(typeof(Factor), "GetItems<>", Default=false)]
    [Buffers("marginal")]
    [Quality(QualityBand.Mature)]
    public static class GetItemsBufferOp2<T>
    {
        public static ArrayType MarginalInit<ArrayType, DistributionType>(ArrayType array, IList<DistributionType> items, IList<int> indices)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>, ICloneable
            where DistributionType : SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            ArrayType result = (ArrayType) array.Clone();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        public static ArrayType Marginal<ArrayType, DistributionType>(ArrayType marginal, DistributionType to_item, DistributionType item, IList<int> indices, int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            ArrayType result = marginal;
            int i = resultIndex;
            DistributionType value = result[indices[i]];
            value.SetToProduct(to_item, item);
            result[indices[i]] = value;
            return result;
        }

        public static DistributionType ItemsAverageConditional<ArrayType, DistributionType>([MatchingIndex] IList<DistributionType> items, [IgnoreDependency] ArrayType array,
                                                                                            [SkipIfAllUniform] ArrayType marginal, IList<int> indices, int resultIndex,
                                                                                            DistributionType result)
            where ArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToRatio(marginal[indices[i]], items[i]);
            return result;
        }
    }
}
