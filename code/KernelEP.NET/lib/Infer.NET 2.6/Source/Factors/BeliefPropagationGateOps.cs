/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Gate.EnterOne{T}(int, T, int)" />, given random arguments to the function.</summary>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "EnterOne<>", null, typeof(int), null, typeof(int))]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterOneOp
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterOne">Incoming message from <c>enterOne</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterOne,selector) p(enterOne,selector) factor(enterOne,selector,value,index)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterOne" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, Discrete selector, TDist value, int index, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableToWeightedSum<TDist>, SettableTo<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (selector == null)
            {
                throw new ArgumentNullException("selector");
            }

            double logProbSum = selector.GetLogProb(index);
            if (logProbSum == 0.0)
            {
                result.SetTo(enterOne);
            }
            else if (double.IsNegativeInfinity(logProbSum))
            {
                result.SetToUniform();
            }
            else
            {
                double logProb = MMath.Log1MinusExp(logProbSum);
                double shift = Math.Max(logProbSum, logProb);

                // Avoid (-Infinity) - (-Infinity)
                if (double.IsNegativeInfinity(shift))
                {
                    throw new AllZeroException();
                }

                TDist uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                result.SetToSum(Math.Exp(logProbSum - shift - enterOne.GetLogAverageOf(value)), enterOne, Math.Exp(logProb - shift + uniform.GetLogNormalizer()), uniform);
            }

            return result;
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Gate.Enter{T}(int, T)" /></description></item><item><description><see cref="Gate.Enter{T}(bool, T)" /></description></item></list>, given random arguments to the function.</summary>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(bool), null)]
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(int), null)]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterOp
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enter">Incoming message from <c>enter</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enter,selector) p(enter,selector) factor(enter,selector,value)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enter" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, Discrete selector, TDist value, TDist result)
            where TDist : SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (enter == null)
            {
                throw new ArgumentNullException("enter");
            }

            if (selector == null)
            {
                throw new ArgumentNullException("selector");
            }

            if (selector.Dimension != enter.Count)
            {
                throw new ArgumentException("selector.Dimension != enter.Count");
            }

            // TODO: use pre-allocated buffers
            double logWeightSum = selector.GetLogProb(0);
            if (!double.IsNegativeInfinity(logWeightSum))
            {
                logWeightSum -= enter[0].GetLogAverageOf(value);
                result.SetTo(enter[0]);
            }

            if (selector.Dimension > 1)
            {
                for (int i = 1; i < selector.Dimension; i++)
                {
                    double logWeight = selector.GetLogProb(i);
                    double shift = Math.Max(logWeightSum, logWeight);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == selector.Dimension - 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double logWeightShifted = logWeight - shift;
                        if (!double.IsNegativeInfinity(logWeightShifted))
                        {
                            logWeightShifted -= enter[i].GetLogAverageOf(value);
                            result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeightShifted), enter[i]);
                            logWeightSum = MMath.LogSumExp(logWeightSum, logWeightShifted + shift);
                        }
                    }
                }
            }

            return result;
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Gate.EnterPartial{T}(int, T, int[])" /></description></item><item><description><see cref="Gate.EnterPartial{T}(bool, T, int[])" /></description></item></list>, given random arguments to the function.</summary>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(int), null, typeof(int[]))]
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(bool), null, typeof(int[]))]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterPartialOp
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterPartial,selector) p(enterPartial,selector) factor(enterPartial,selector,value,indices)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartial" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="selector" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] Discrete selector, TDist value, int[] indices, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (enterPartial == null)
            {
                throw new ArgumentNullException("enterPartial");
            }

            if (selector == null)
            {
                throw new ArgumentNullException("selector");
            }

            if (indices == null)
            {
                throw new ArgumentNullException("indices");
            }

            if (indices.Length != enterPartial.Count)
            {
                throw new ArgumentException("indices.Length != enterPartial.Count");
            }

            if (selector.Dimension < enterPartial.Count)
            {
                throw new ArgumentException("selector.Dimension < enterPartial.Count");
            }

            if (indices.Length == 0)
            {
                throw new ArgumentException("indices.Length == 0");
            }

            // TODO: use pre-allocated buffers
            double logProbSum = selector.GetLogProb(indices[0]);
            double logWeightSum = logProbSum;
            if (!double.IsNegativeInfinity(logWeightSum))
            {
                logWeightSum -= enterPartial[0].GetLogAverageOf(value);
                result.SetTo(enterPartial[0]);
            }

            if (indices.Length > 1)
            {
                for (int i = 1; i < indices.Length; i++)
                {
                    double logProb = selector.GetLogProb(indices[i]);
                    logProbSum += logProb;
                    double shift = Math.Max(logWeightSum, logProb);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == selector.Dimension - 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double logWeightShifted = logProb - shift;
                        if (!double.IsNegativeInfinity(logWeightShifted))
                        {
                            logWeightShifted -= enterPartial[i].GetLogAverageOf(value);
                            result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeightShifted), enterPartial[i]);
                            logWeightSum = MMath.LogSumExp(logWeightSum, logWeightShifted + shift);
                        }
                    }
                }
            }

            if (indices.Length < selector.Dimension)
            {
                double logProb = MMath.Log1MinusExp(logProbSum);
                double shift = Math.Max(logWeightSum, logProb);
                if (double.IsNegativeInfinity(shift))
                {
                    throw new AllZeroException();
                }

                var uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                double logWeight = logProb + uniform.GetLogNormalizer();
                result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeight - shift), uniform);
            }

            return result;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterPartial,selector) p(enterPartial,selector) factor(enterPartial,selector,value,indices)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartial" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="selector" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] Bernoulli selector, TDist value, int[] indices, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (enterPartial == null)
            {
                throw new ArgumentNullException("enterPartial");
            }

            if (indices == null)
            {
                throw new ArgumentNullException("indices");
            }

            if (indices.Length != enterPartial.Count)
            {
                throw new ArgumentException("indices.Length != enterPartial.Count");
            }

            if (2 < enterPartial.Count)
            {
                throw new ArgumentException("enterPartial.Count should be 2 or 1");
            }

            if (indices.Length == 0)
            {
                throw new ArgumentException("indices.Length == 0");
            }

            // TODO: use pre-allocated buffers
            double logProbSum = (indices[0] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
            double logWeightSum = logProbSum;
            if (!double.IsNegativeInfinity(logProbSum))
            {
                logWeightSum -= enterPartial[0].GetLogAverageOf(value);
                result.SetTo(enterPartial[0]);
            }

            if (indices.Length > 1)
            {
                for (int i = 1; i < indices.Length; i++)
                {
                    double logProb = (indices[i] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
                    logProbSum += logProb;
                    double shift = Math.Max(logWeightSum, logProb);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double logWeightShifted = logProb - shift;
                        if (!double.IsNegativeInfinity(logWeightShifted))
                        {
                            logWeightShifted -= enterPartial[i].GetLogAverageOf(value);
                            result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeightShifted), enterPartial[i]);
                            logWeightSum = MMath.LogSumExp(logWeightSum, logWeightShifted + shift);
                        }
                    }
                }
            }

            if (indices.Length < 2)
            {
                double logProb = MMath.Log1MinusExp(logProbSum);
                double shift = Math.Max(logWeightSum, logProb);
                if (double.IsNegativeInfinity(shift))
                {
                    throw new AllZeroException();
                }

                var uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                double logWeight = logProb + uniform.GetLogNormalizer();
                result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeight - shift), uniform);
            }

            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.EnterPartialTwo{T}(bool, bool, T, int[])" />, given random arguments to the function.</summary>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "EnterPartialTwo<>")]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterPartialTwoOp
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterPartialTwo">Incoming message from <c>enterPartialTwo</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterPartialTwo,case0,case1) p(enterPartialTwo,case0,case1) factor(enterPartialTwo,case0,case1,value,indices)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartialTwo" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartialTwo, Bernoulli case0, Bernoulli case1, TDist value, int[] indices, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (enterPartialTwo == null)
            {
                throw new ArgumentNullException("enterPartialTwo");
            }

            if (indices == null)
            {
                throw new ArgumentNullException("indices");
            }

            if (indices.Length != enterPartialTwo.Count)
            {
                throw new ArgumentException("indices.Length != enterPartialTwo.Count");
            }

            if (2 < enterPartialTwo.Count)
            {
                throw new ArgumentException("enterPartialTwo.Count should be 2 or 1");
            }

            if (indices.Length == 0)
            {
                throw new ArgumentException("indices.Length == 0");
            }

            // TODO: use pre-allocated buffers
            double logProb0 = (indices[0] == 0 ? case0 : case1).LogOdds;
            double logProb1 = (indices[0] == 0 ? case1 : case0).LogOdds;
            double shift = Math.Max(logProb0, logProb1);

            // Avoid (-Infinity) - (-Infinity)
            if (double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
            }

            if (indices.Length > 1)
            {
                result.SetToSum(
                    Math.Exp(logProb0 - shift - value.GetLogAverageOf(enterPartialTwo[0])),
                    enterPartialTwo[0],
                    Math.Exp(logProb1 - shift - value.GetLogAverageOf(enterPartialTwo[1])),
                    enterPartialTwo[1]);
            }
            else
            {
                TDist uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                result.SetToSum(
                    Math.Exp(logProb0 - shift - value.GetLogAverageOf(enterPartialTwo[0])),
                    enterPartialTwo[0],
                    Math.Exp(logProb1 - shift + uniform.GetLogNormalizer()),
                    uniform);
            }

            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.Exit{T}(bool[], T[])" />, given random arguments to the function.</summary>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable exiting the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "Exit<>")]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateExitOp
    {
        /// <summary>EP message to <c>exit</c>.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exit</c> as the random arguments are varied. The formula is <c>proj[p(exit) sum_(cases,values) p(cases,values) factor(exit,cases,values)]/p(exit)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional<TDist>(IList<Bernoulli> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>, SettableToWeightedSum<TDist>
        {
            if (cases == null)
            {
                throw new ArgumentNullException("cases");
            }

            if (values == null)
            {
                throw new ArgumentNullException("values");
            }

            if (cases.Count != values.Count)
            {
                throw new ArgumentException("cases.Count != values.Count");
            }

            if (cases.Count == 0)
            {
                throw new ArgumentException("cases.Count == 0");
            }

            if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                double logResultProb = cases[0].LogOdds;
                if (double.IsNaN(logResultProb))
                {
                    throw new AllZeroException();
                }

                // TODO: use pre-allocated buffer
                int resultIndex = 0;
                for (int i = 1; i < cases.Count; i++)
                {
                    double logProb = cases[i].LogOdds;
                    double shift = Math.Max(logResultProb, logProb);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double weight1 = Math.Exp(logResultProb - shift);
                        double weight2 = Math.Exp(logProb - shift);
                        if (weight2 > 0)
                        {
                            if (weight1 == 0)
                            {
                                resultIndex = i;
                                logResultProb = logProb;
                            }
                            else
                            {
                                if (resultIndex >= 0)
                                {
                                    result.SetTo(values[resultIndex]);
                                    resultIndex = -1;
                                }

                                result.SetToSum(weight1, result, weight2, values[i]);
                                logResultProb = MMath.LogSumExp(logResultProb, logProb);
                            }
                        }
                    }
                }

                if (resultIndex >= 0)
                {
                    // result is simply values[resultIndex]
                    return values[resultIndex];
                }
            }

            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.ExitTwo{T}(bool, bool, T[])" />, given random arguments to the function.</summary>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable exiting the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "ExitTwo<>")]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateExitTwoOp
    {
        /// <summary>EP message to <c>exitTwo</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>.</param>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exitTwo</c> as the random arguments are varied. The formula is <c>proj[p(exitTwo) sum_(case0,case1,values) p(case0,case1,values) factor(exitTwo,case0,case1,values)]/p(exitTwo)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitTwoAverageConditional<TDist>(
            TDist exitTwo, Bernoulli case0, Bernoulli case1, [SkipIfAllUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (values == null)
            {
                throw new ArgumentNullException("values");
            }

            if (values.Count != 2)
            {
                throw new ArgumentException("values.Count != 2");
            }

            double logProb0 = case0.LogOdds;
            double logProb1 = case1.LogOdds;
            double shift = Math.Max(logProb0, logProb1);

            // Avoid (-Infinity) - (-Infinity)
            if (double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
            }

            result.SetToSum(
                Math.Exp(logProb0 - shift - exitTwo.GetLogAverageOf(values[0])),
                values[0],
                Math.Exp(logProb1 - shift - exitTwo.GetLogAverageOf(values[1])),
                values[1]);
            return result;
        }
    }
}
