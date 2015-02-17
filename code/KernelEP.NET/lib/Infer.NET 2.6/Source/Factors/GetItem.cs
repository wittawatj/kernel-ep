// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.GetItem{T}(IList{T}, int)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Factor), "GetItem<>")]
    [Quality(QualityBand.Mature)]
    public static class GetItemOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item,array) p(item,array) factor(item,array,index))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(T item, IList<T> array, int index)
        {
            IEqualityComparer<T> equalityComparer = Utils.Util.GetEqualityComparer<T>();
            return equalityComparer.Equals(item, array[index]) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item,array) p(item,array) factor(item,array,index) / sum_item p(item) messageTo(item))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(T item, IList<T> array, int index)
        {
            return LogAverageFactor(item, array, index);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(T item, IList<T> array, int index)
        {
            return LogAverageFactor(item, array, index);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="to_item">Outgoing message to <c>item</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item) p(item) factor(item,array,index))</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogAverageFactor<Distribution>(Distribution item, [Fresh] Distribution to_item)
            where Distribution : CanGetLogAverageOf<Distribution>
        {
            return to_item.GetLogAverageOf(item);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item,array) p(item,array) factor(item,array,index))</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogAverageFactor<Distribution>(T item, IList<Distribution> array, int index)
            where Distribution : CanGetLogProb<T>
        {
            return array[index].GetLogProb(item);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item) p(item) factor(item,array,index) / sum_item p(item) messageTo(item))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<Distribution>(Distribution item) where Distribution : IDistribution<T>
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item,array) p(item,array) factor(item,array,index) / sum_item p(item) messageTo(item))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogEvidenceRatio<Distribution>(T item, IList<Distribution> array, int index)
            where Distribution : CanGetLogProb<T>
        {
            return LogAverageFactor(item, array, index);
        }

        /// <summary>EP message to <c>item</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>item</c> as the random arguments are varied. The formula is <c>proj[p(item) sum_(array) p(array) factor(item,array,index)]/p(item)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageConditional<Distribution>([SkipIfAllUniform] IList<Distribution> array, int index, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index]);
            return result;
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static Distribution ItemAverageConditionalInit<Distribution>([IgnoreDependency] IList<Distribution> array)
            where Distribution : ICloneable
        {
            return (Distribution)array[0].Clone();
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(item) p(item) factor(item,array,index)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="item" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>([SkipIfUniform] Distribution item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index];
            value.SetTo(item);
            result[index] = value;
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(item) p(item) factor(item,array,index)]/p(array)</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>(T item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index];
            value.Point = item;
            result[index] = value;
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(item,array,index))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>item</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>item</c> as the random arguments are varied. The formula is <c>proj[sum_(array) p(array) factor(item,array,index)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageLogarithm<Distribution>([SkipIfAllUniform] IList<Distribution> array, int index, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index]);
            return result;
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static Distribution ItemAverageLogarithmInit<Distribution>([IgnoreDependency] IList<Distribution> array)
            where Distribution : ICloneable
        {
            return (Distribution)array[0].Clone();
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>item</c> integrated out. The formula is <c>sum_item p(item) factor(item,array,index)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="item" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>([SkipIfUniform] Distribution item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            Distribution value = result[index];
            value.SetTo(item);
            result[index] = value;
            return result;
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>item</c> integrated out. The formula is <c>sum_item p(item) factor(item,array,index)</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>(T item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index];
            value.Point = item;
            result[index] = value;
            return result;
        }
    }
}
