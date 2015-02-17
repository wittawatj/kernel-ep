// (C) Copyright 2009-2010 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Vector.Subvector(Vector, int, int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Vector), "Subvector", typeof(Vector), typeof(int), typeof(int))]
    [Buffers("SourceMean", "SourceVariance")]
    [Quality(QualityBand.Stable)]
    public static class SubvectorOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="subvector">Constant value for <c>subvector</c>.</param>
        /// <param name="source">Constant value for <c>source</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(subvector,source,startIndex,count))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector subvector, Vector source, int startIndex)
        {
            for (int i = 0; i < subvector.Count; i++)
            {
                if (subvector[i] != source[startIndex + i])
                    return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="subvector">Constant value for <c>subvector</c>.</param>
        /// <param name="source">Constant value for <c>source</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(subvector,source,startIndex,count))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector subvector, Vector source, int startIndex)
        {
            return LogAverageFactor(subvector, source, startIndex);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="subvector">Constant value for <c>subvector</c>.</param>
        /// <param name="source">Constant value for <c>source</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(subvector,source,startIndex,count))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Vector subvector, Vector source, int startIndex)
        {
            return LogAverageFactor(subvector, source, startIndex);
        }

        /// <summary>Initialize the buffer <c>SourceVariance</c>.</summary>
        /// <param name="Source">Incoming message from <c>source</c>.</param>
        /// <returns>Initial value of buffer <c>SourceVariance</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static PositiveDefiniteMatrix SourceVarianceInit([IgnoreDependency] VectorGaussian Source)
        {
            return new PositiveDefiniteMatrix(Source.Dimension, Source.Dimension);
        }

        /// <summary>Update the buffer <c>SourceVariance</c>.</summary>
        /// <param name="Source">Incoming message from <c>source</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Source" /> is not a proper distribution.</exception>
        public static PositiveDefiniteMatrix SourceVariance([Proper] VectorGaussian Source, PositiveDefiniteMatrix result)
        {
            return Source.GetVariance(result);
        }

        /// <summary>Initialize the buffer <c>SourceMean</c>.</summary>
        /// <param name="Source">Incoming message from <c>source</c>.</param>
        /// <returns>Initial value of buffer <c>SourceMean</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Vector SourceMeanInit([IgnoreDependency] VectorGaussian Source)
        {
            return Vector.Zero(Source.Dimension);
        }

        /// <summary>Update the buffer <c>SourceMean</c>.</summary>
        /// <param name="Source">Incoming message from <c>source</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SourceVariance">Buffer <c>SourceVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Source" /> is not a proper distribution.</exception>
        public static Vector SourceMean([Proper] VectorGaussian Source, [Fresh] PositiveDefiniteMatrix SourceVariance, Vector result)
        {
            return Source.GetMean(result, SourceVariance);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="subvector">Constant value for <c>subvector</c>.</param>
        /// <param name="SourceMean">Buffer <c>SourceMean</c>.</param>
        /// <param name="SourceVariance">Buffer <c>SourceVariance</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(subvector,source,startIndex,count))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector subvector, [Fresh] Vector SourceMean, [Fresh] PositiveDefiniteMatrix SourceVariance, int startIndex)
        {
            double sum = 0.0;
            for (int i = startIndex; i < SourceMean.Count; i++)
            {
                sum += Gaussian.GetLogProb(subvector[i], SourceMean[i], SourceVariance[i, i]);
            }
            return sum;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="subvector">Constant value for <c>subvector</c>.</param>
        /// <param name="SourceMean">Buffer <c>SourceMean</c>.</param>
        /// <param name="SourceVariance">Buffer <c>SourceVariance</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(subvector,source,startIndex,count))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector subvector, [Fresh] Vector SourceMean, [Fresh] PositiveDefiniteMatrix SourceVariance, int startIndex)
        {
            return LogAverageFactor(subvector, SourceMean, SourceVariance, startIndex);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="subvector">Incoming message from <c>subvector</c>.</param>
        /// <param name="to_subvector">Outgoing message to <c>subvector</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(subvector) p(subvector) factor(subvector,source,startIndex,count))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(VectorGaussian subvector, [Fresh] VectorGaussian to_subvector)
        {
            return to_subvector.GetLogAverageOf(subvector);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="subvector">Incoming message from <c>subvector</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(subvector) p(subvector) factor(subvector,source,startIndex,count) / sum_subvector p(subvector) messageTo(subvector))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian subvector)
        {
            return 0.0;
        }

        /// <summary />
        /// <param name="count">Constant value for <c>count</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian SubvectorAverageConditionalInit(int count)
        {
            return new VectorGaussian(count);
        }

        /// <summary />
        /// <param name="count">Constant value for <c>count</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian SubvectorAverageLogarithmInit(int count)
        {
            return new VectorGaussian(count);
        }

        /// <summary>EP message to <c>subvector</c>.</summary>
        /// <param name="SourceMean">Buffer <c>SourceMean</c>.</param>
        /// <param name="SourceVariance">Buffer <c>SourceVariance</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>subvector</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SubvectorAverageConditional(
            [Fresh] Vector SourceMean, [Fresh] PositiveDefiniteMatrix SourceVariance, int startIndex, VectorGaussian result)
        {
            PositiveDefiniteMatrix subVariance = new PositiveDefiniteMatrix(result.Dimension, result.Dimension);
            subVariance.SetToSubmatrix(SourceVariance, startIndex, startIndex);
            Vector subMean = Vector.Zero(result.Dimension);
            subMean.SetToSubvector(SourceMean, startIndex);
            result.SetMeanAndVariance(subMean, subVariance);
            return result;
        }

        /// <summary>EP message to <c>source</c>.</summary>
        /// <param name="subvector">Incoming message from <c>subvector</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>source</c> as the random arguments are varied. The formula is <c>proj[p(source) sum_(subvector) p(subvector) factor(subvector,source,startIndex,count)]/p(source)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="subvector" /> is not a proper distribution.</exception>
        public static VectorGaussian SourceAverageConditional([SkipIfUniform] VectorGaussian subvector, int startIndex, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision.SetSubvector(startIndex, subvector.MeanTimesPrecision);
            result.Precision.SetAllElementsTo(0.0);
            result.Precision.SetSubmatrix(startIndex, startIndex, subvector.Precision);
            return result;
        }

        /// <summary>EP message to <c>source</c>.</summary>
        /// <param name="subvector">Constant value for <c>subvector</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>source</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SourceAverageConditional(Vector subvector, int startIndex, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision.SetSubvector(startIndex, subvector);
            result.Precision.SetAllElementsTo(0.0);
            int dim = result.Dimension;
            for (int i = startIndex; i < dim; i++)
            {
                result.Precision[i, i] = Double.PositiveInfinity;
            }
            return result;
        }

        //-- VMP ---------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(subvector,source,startIndex,count))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>subvector</c>.</summary>
        /// <param name="SourceMean">Buffer <c>SourceMean</c>.</param>
        /// <param name="SourceVariance">Buffer <c>SourceVariance</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>subvector</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SubvectorAverageLogarithm([Fresh] Vector SourceMean, [Fresh] PositiveDefiniteMatrix SourceVariance, int startIndex, VectorGaussian result)
        {
            return SubvectorAverageConditional(SourceMean, SourceVariance, startIndex, result);
        }

        /// <summary>VMP message to <c>source</c>.</summary>
        /// <param name="subvector">Incoming message from <c>subvector</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>source</c> with <c>subvector</c> integrated out. The formula is <c>sum_subvector p(subvector) factor(subvector,source,startIndex,count)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="subvector" /> is not a proper distribution.</exception>
        public static VectorGaussian SourceAverageLogarithm([SkipIfUniform] VectorGaussian subvector, int startIndex, VectorGaussian result)
        {
            return SourceAverageConditional(subvector, startIndex, result);
        }

        /// <summary>VMP message to <c>source</c>.</summary>
        /// <param name="subvector">Constant value for <c>subvector</c>.</param>
        /// <param name="startIndex">Constant value for <c>startIndex</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>source</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SourceAverageLogarithm(Vector subvector, int startIndex, VectorGaussian result)
        {
            return SourceAverageConditional(subvector, startIndex, result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.GetItem{T}(IList{T}, int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "GetItem<>", typeof(double), typeof(Vector), typeof(int))]
    [Buffers("ArrayMean", "ArrayVariance")]
    [Quality(QualityBand.Preview)]
    public static class VectorElementOp
    {
        /// <summary>Initialize the buffer <c>ArrayVariance</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns>Initial value of buffer <c>ArrayVariance</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static PositiveDefiniteMatrix ArrayVarianceInit([IgnoreDependency] VectorGaussian array)
        {
            return new PositiveDefiniteMatrix(array.Dimension, array.Dimension);
        }

        /// <summary>Update the buffer <c>ArrayVariance</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        public static PositiveDefiniteMatrix ArrayVariance([Proper] VectorGaussian array, PositiveDefiniteMatrix result)
        {
            return array.GetVariance(result);
        }

        /// <summary>Initialize the buffer <c>ArrayMean</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns>Initial value of buffer <c>ArrayMean</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Vector ArrayMeanInit([IgnoreDependency] VectorGaussian array)
        {
            return Vector.Zero(array.Dimension);
        }

        /// <summary>Update the buffer <c>ArrayMean</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="ArrayVariance">Buffer <c>ArrayVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        public static Vector ArrayMean([Proper] VectorGaussian array, [Fresh] PositiveDefiniteMatrix ArrayVariance, Vector result)
        {
            return array.GetMean(result, ArrayVariance);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Constant value for <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="ArrayMean">Buffer <c>ArrayMean</c>.</param>
        /// <param name="ArrayVariance">Buffer <c>ArrayVariance</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(array) p(array) factor(item,array,index))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(
            double item, [SkipIfUniform] VectorGaussian array, [Fresh] Vector ArrayMean, [Fresh] PositiveDefiniteMatrix ArrayVariance, int index)
        {
            Gaussian to_item = ItemAverageConditional(array, ArrayMean, ArrayVariance, index);
            return to_item.GetLogProb(item);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="item">Constant value for <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="ArrayMean">Buffer <c>ArrayMean</c>.</param>
        /// <param name="ArrayVariance">Buffer <c>ArrayVariance</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(array) p(array) factor(item,array,index))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(
            double item, [SkipIfUniform] VectorGaussian array, [Fresh] Vector ArrayMean, [Fresh] PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return LogAverageFactor(item, array, ArrayMean, ArrayVariance, index);
        }

        /// <summary>EP message to <c>item</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="ArrayMean">Buffer <c>ArrayMean</c>.</param>
        /// <param name="ArrayVariance">Buffer <c>ArrayVariance</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>The outgoing EP message to the <c>item</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>item</c> as the random arguments are varied. The formula is <c>proj[p(item) sum_(array) p(array) factor(item,array,index)]/p(item)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        public static Gaussian ItemAverageConditional(
            [SkipIfUniform] VectorGaussian array, [Fresh] Vector ArrayMean, [Fresh] PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return new Gaussian(ArrayMean[index], ArrayVariance[index, index]);
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gaussian ItemAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gaussian ItemAverageLogarithmInit()
        {
            return Gaussian.Uniform();
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
        public static VectorGaussian ArrayAverageConditional([SkipIfUniform] Gaussian item, int index, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision[index] = item.MeanTimesPrecision;
            result.Precision.SetAllElementsTo(0.0);
            result.Precision[index, index] = item.Precision;
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="item">Constant value for <c>item</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian ArrayAverageConditional(double item, int index, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision[index] = item;
            result.Precision.SetAllElementsTo(0.0);
            result.Precision[index, index] = Double.PositiveInfinity;
            return result;
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="item">Constant value for <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="ArrayMean">Buffer <c>ArrayMean</c>.</param>
        /// <param name="ArrayVariance">Buffer <c>ArrayVariance</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            double item, [SkipIfUniform] VectorGaussian array, [Fresh] Vector ArrayMean, [Fresh] PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return LogAverageFactor(item, array, ArrayMean, ArrayVariance, index);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="item">Incoming message from <c>item</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor(Gaussian item, VectorGaussian array)
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>item</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="ArrayMean">Buffer <c>ArrayMean</c>.</param>
        /// <param name="ArrayVariance">Buffer <c>ArrayVariance</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <returns>The outgoing VMP message to the <c>item</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>item</c> as the random arguments are varied. The formula is <c>proj[sum_(array) p(array) factor(item,array,index)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        public static Gaussian ItemAverageLogarithm(
            [SkipIfUniform] VectorGaussian array, [Fresh] Vector ArrayMean, [Fresh] PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return ItemAverageConditional(array, ArrayMean, ArrayVariance, index);
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
        public static VectorGaussian ArrayAverageLogarithm([SkipIfUniform] Gaussian item, int index, VectorGaussian result)
        {
            return ArrayAverageConditional(item, index, result);
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="item">Constant value for <c>item</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian ArrayAverageLogarithm(double item, int index, VectorGaussian result)
        {
            return ArrayAverageConditional(item, index, result);
        }
    }
}
