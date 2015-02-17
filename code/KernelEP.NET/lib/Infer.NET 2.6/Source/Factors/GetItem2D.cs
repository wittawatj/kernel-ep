// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.GetItem2D{T}(T[,], int, int)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Factor), "GetItem2D<>")]
    [Quality(QualityBand.Stable)]
    public static class GetItem2DOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="to_item">Outgoing message to <c>item</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item) p(item) factor(item,array,index1,index2))</c>.</para>
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
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item,array) p(item,array) factor(item,array,index1,index2))</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogAverageFactor<Distribution>(T item, IArray2D<Distribution> array, int index1, int index2)
            where Distribution : CanGetLogProb<T>
        {
            return array[index1, index2].GetLogProb(item);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item) p(item) factor(item,array,index1,index2) / sum_item p(item) messageTo(item))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<Distribution>(Distribution item)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(item,array) p(item,array) factor(item,array,index1,index2) / sum_item p(item) messageTo(item))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogEvidenceRatio<Distribution>(T item, IArray2D<Distribution> array, int index1, int index2)
            where Distribution : CanGetLogProb<T>
        {
            return LogAverageFactor(item, array, index1, index2);
        }

        /// <summary>EP message to <c>item</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>item</c> as the random arguments are varied. The formula is <c>proj[p(item) sum_(array) p(array) factor(item,array,index1,index2)]/p(item)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageConditional<Distribution>([SkipIfAllUniform] IArray2D<Distribution> array, int index1, int index2, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index1, index2]);
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(item) p(item) factor(item,array,index1,index2)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="item" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>(
            [SkipIfUniform] Distribution item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index1, index2];
            value.SetTo(item);
            result[index1, index2] = value;
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(item) p(item) factor(item,array,index1,index2)]/p(array)</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>(
            T item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index1, index2];
            value.Point = item;
            result[index1, index2] = value;
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(item,array,index1,index2))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>item</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>item</c> as the random arguments are varied. The formula is <c>proj[sum_(array) p(array) factor(item,array,index1,index2)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageLogarithm<Distribution>(
            [SkipIfAllUniform] IArray2D<Distribution> array, int index1, int index2, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index1, index2]);
            return result;
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>item</c> integrated out. The formula is <c>sum_item p(item) factor(item,array,index1,index2)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="item" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>(
            [SkipIfUniform] Distribution item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            Distribution value = result[index1, index2];
            value.SetTo(item);
            result[index1, index2] = value;
            return result;
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="index1">Constant value for <c>index1</c>.</param>
        /// <param name="index2">Constant value for <c>index2</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>item</c> integrated out. The formula is <c>sum_item p(item) factor(item,array,index1,index2)</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>(
            T item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index1, index2];
            value.Point = item;
            result[index1, index2] = value;
            return result;
        }
    }
}
