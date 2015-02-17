// (C) Copyright 2008 Microsoft Research Cambridge

#define UseRatioDir

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Gate.EnterPartial{T}(int, T, int[])" /></description></item><item><description><see cref="Gate.EnterPartial{T}(bool, T, int[])" /></description></item></list>, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable entering the gate.</typeparam>
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(int), null, typeof(int[]))]
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(bool), null, typeof(int[]))]
    [Quality(QualityBand.Mature)]
    public static class GateEnterPartialOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterPartial,selector,value,indices))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterPartial,selector,value,indices))</c>.</para>
        /// </remarks>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>enterPartial</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enterPartial</c> as the random arguments are varied. The formula is <c>proj[p(enterPartial) sum_(value) p(value) factor(enterPartial,selector,value,indices)]/p(enterPartial)</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialAverageConditional<TValue, TResultList>([IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <summary>Initialize the buffer <c>enterPartial</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="factory" />
        /// <returns>Initial value of buffer <c>enterPartial</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        /// <typeparam name="TArray">The type of an array that can be produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static TArray EnterPartialInit<TValue, TArray>([IgnoreDependency] TValue value, int[] indices, IArrayFactory<TValue, TArray> factory)
            where TValue : ICloneable
        {
            return factory.CreateArray(indices.Length, i => (TValue)value.Clone());
        }

        /// <summary>EP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Discrete SelectorAverageConditional(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>EP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Bernoulli SelectorAverageConditional(Bernoulli result)
        {
            result.SetToUniform();
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
            [SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] Discrete selector, TDist value, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (selector.Dimension < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                // TODO: use pre-allocated buffers
                double logProbSum = selector.GetLogProb(indices[0]);
                if (!double.IsNegativeInfinity(logProbSum))
                {
                    try
                    {
                        result.SetToProduct(value, enterPartial[0]);
                    }
                    catch (AllZeroException)
                    {
                        logProbSum = double.NegativeInfinity;
                    }
                }
                if (indices.Length > 1)
                {
                    TDist product = (TDist)value.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        double logProb = selector.GetLogProb(indices[i]);
                        double shift = Math.Max(logProbSum, logProb);
                        // avoid (-Infinity) - (-Infinity)
                        if (Double.IsNegativeInfinity(shift))
                        {
                            if (i == selector.Dimension - 1)
                            {
                                throw new AllZeroException();
                            }
                            // do nothing
                        }
                        else
                        {
                            double productWeight = Math.Exp(logProb - shift);
                            if (productWeight > 0)
                            {
                                try
                                {
                                    product.SetToProduct(value, enterPartial[i]);
                                }
                                catch (AllZeroException)
                                {
                                    productWeight = 0;
                                }
                                if (productWeight > 0)
                                {
                                    result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                                    logProbSum = MMath.LogSumExp(logProbSum, logProb);
                                }
                            }
                        }
                    }
                }
                if (indices.Length < selector.Dimension)
                {
                    double logProb = MMath.Log1MinusExp(logProbSum);
                    double shift = Math.Max(logProbSum, logProb);
                    if (Double.IsNegativeInfinity(shift))
                        throw new AllZeroException();
                    result.SetToSum(Math.Exp(logProbSum - shift), result, Math.Exp(logProb - shift), value);
                }
                result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Constant value for <c>selector</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterPartial) p(enterPartial) factor(enterPartial,selector,value,indices)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartial" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, int selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if (selector == indices[i])
                    {
                        result.SetTo(enterPartial[i]);
                        break;
                    }
                }
                return result;
            }
        }

#if false
		public static TDist ValueAverageConditional<TDist>(
			IList<T> enterPartial,
			int selector, int[] indices, TDist result)
			where TDist : IDistribution<T>
		{
			if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
			if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
			else {
				result.SetToUniform();
				for (int i = 0; i < indices.Length; i++) {
					if (selector == indices[i]) {
						result.Point = enterPartial[i];
						break;
					}
				}
				return result;
			}
		}
#endif

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
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                // TODO: use pre-allocated buffers
                double logProbSum = (indices[0] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
                if (!double.IsNegativeInfinity(logProbSum))
                {
                    result.SetToProduct(value, enterPartial[0]);
                }
                if (indices.Length > 1)
                {
                    TDist product = (TDist)value.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        double logProb = (indices[i] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
                        double shift = Math.Max(logProbSum, logProb);
                        // avoid (-Infinity) - (-Infinity)
                        if (Double.IsNegativeInfinity(shift))
                        {
                            if (i == 1)
                            {
                                throw new AllZeroException();
                            }
                            // do nothing
                        }
                        else
                        {
                            double productWeight = Math.Exp(logProb - shift);
                            if (productWeight > 0)
                            {
                                product.SetToProduct(value, enterPartial[i]);
                                result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                                logProbSum = MMath.LogSumExp(logProbSum, logProb);
                            }
                        }
                    }
                }
                if (indices.Length < 2)
                {
                    double logProb = MMath.Log1MinusExp(logProbSum);
                    double shift = Math.Max(logProbSum, logProb);
                    if (Double.IsNegativeInfinity(shift))
                        throw new AllZeroException();
                    result.SetToSum(Math.Exp(logProbSum - shift), result, Math.Exp(logProb - shift), value);
                }
                result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
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
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, bool selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                int caseNumber = selector ? 0 : 1;
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if (caseNumber == indices[i])
                    {
                        result.SetTo(enterPartial[i]);
                        break;
                    }
                }
                return result;
            }
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterPartial,selector) p(enterPartial,selector) factor(enterPartial,selector,value,indices)]/p(value)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            IList<T> enterPartial, bool selector, int[] indices, TDist result)
            where TDist : IDistribution<T>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                int caseNumber = selector ? 0 : 1;
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if (caseNumber == indices[i])
                    {
                        result.Point = enterPartial[i];
                        break;
                    }
                }
                return result;
            }
        }

#if false
    /// <summary>
    /// EP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'cases' conditioned on the given values.
    /// </para></remarks>
		[Skip]
		public static BernoulliList CasesAverageConditional<BernoulliList>(BernoulliList result)
			where BernoulliList : SettableToUniform
		{
			result.SetToUniform();
			return result;
		}
		/// <summary>
		/// EP message to 'value'
		/// </summary>
		/// <param name="enterPartial">Incoming message from 'enterPartial'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="cases">Incoming message from 'cases'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
		/// <param name="value">Incoming message from 'value'.</param>
		/// <param name="indices">Constant value for 'indices'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
		/// The formula is <c>proj[p(value) sum_(enterPartial,cases) p(enterPartial,cases) factor(enterPartial,cases,value,indices)]/p(value)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enterPartial"/> is not a proper distribution</exception>
		/// <exception cref="ImproperMessageException"><paramref name="cases"/> is not a proper distribution</exception>
		public static TDist ValueAverageConditional<TDist>([SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] IList<Bernoulli> cases, TDist value, int[] indices, TDist result)
			where TDist : IDistribution<T>, SettableToProduct<TDist>,
								SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
		{
			if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
			if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
			if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
			else {
				// TODO: use pre-allocated buffers
				double logProbSum = cases[indices[0]].LogOdds;
				if (!double.IsNegativeInfinity(logProbSum)) {
					result.SetToProduct(value, enterPartial[0]);
				}
				if (indices.Length > 1) {
					TDist product = (TDist)value.Clone();
					for (int i = 1; i < indices.Length; i++) {
						double logProb = cases[indices[i]].LogOdds;
						double shift = Math.Max(logProbSum, logProb);
						// avoid (-Infinity) - (-Infinity)
						if (Double.IsNegativeInfinity(shift)) {
							if (i == cases.Count - 1) {
								throw new AllZeroException();
							}
							// do nothing
						} else {
							double productWeight = Math.Exp(logProb - shift);
							if (productWeight > 0) {
								product.SetToProduct(value, enterPartial[i]);
								result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
								logProbSum = MMath.LogSumExp(logProbSum, logProb);
							}
						}
					}
				}
				if (indices.Length < cases.Count) {
					double logProb = MMath.Log1MinusExp(logProbSum);
					double shift = Math.Max(logProbSum, logProb);
					if (Double.IsNegativeInfinity(shift)) throw new AllZeroException();
					result.SetToSum(Math.Exp(logProbSum - shift), result, Math.Exp(logProb - shift), value);
				}
				if (GateEnterOp<T>.ForceProper && (result is Gaussian)) {
					Gaussian r = (Gaussian)(object)result;
					r.SetToRatioProper(r, (Gaussian)(object)value);
					result = (TDist)(object)r;
				} else {
					result.SetToRatio(result, value);
				}
			}
			return result;
		}
		/// <summary>
		/// EP message to 'value'
		/// </summary>
		/// <param name="enterPartial">Incoming message from 'enterPartial'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="cases">Constant value for 'cases'.</param>
		/// <param name="indices">Constant value for 'indices'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
		/// The formula is <c>proj[p(value) sum_(enterPartial,cases) p(enterPartial,cases) factor(enterPartial,cases,value,indices)]/p(value)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enterPartial"/> is not a proper distribution</exception>
		public static TDist ValueAverageConditional<TDist>(
			[SkipIfAllUniform] IList<TDist> enterPartial,
			IList<bool> cases, int[] indices, TDist result)
			where TDist : IDistribution<T>, SettableTo<TDist>
		{
			if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
			if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
			if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
			else {
				result.SetToUniform();
				for (int i = 0; i < indices.Length; i++) {
					if (cases[indices[i]]) {
						result.SetTo(enterPartial[i]);
						break;
					}
				}
				return result;
			}
		}
		public static TDist ValueAverageConditional<TDist>(
			IList<T> enterPartial,
			IList<bool> cases, int[] indices, TDist result)
			where TDist : IDistribution<T>
		{
			if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
			if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
			if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
			else {
				for (int i = 0; i < indices.Length; i++) {
					if (cases[indices[i]]) {
						result.Point = enterPartial[i];
						break;
					}
				}
				return result;
			}
		}
#endif

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterPartial,selector,value,indices))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>enterPartial</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enterPartial</c> as the random arguments are varied. The formula is <c>proj[sum_(value) p(value) factor(enterPartial,selector,value,indices)]</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialAverageLogarithm<TValue, TResultList>(
            [IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Constant value for <c>selector</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>value</c> with <c>enterPartial</c> integrated out. The formula is <c>sum_enterPartial p(enterPartial) factor(enterPartial,selector,value,indices)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartial" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, int selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            return ValueAverageConditional(enterPartial, selector, indices, result);
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>value</c>. Because the factor is deterministic, <c>enterPartial</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(selector) p(selector) log(sum_enterPartial p(enterPartial) factor(enterPartial,selector,value,indices)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartial" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam> 
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, Discrete selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (selector.Dimension < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                double scale = selector[indices[0]];
                result.SetToPower(enterPartial[0], scale);
                if (indices.Length > 1)
                {
                    // TODO: use pre-allocated buffer
                    TDist power = (TDist)result.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        scale = selector[indices[i]];
                        power.SetToPower(enterPartial[i], scale);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>value</c>. Because the factor is deterministic, <c>enterPartial</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(selector) p(selector) log(sum_enterPartial p(enterPartial) factor(enterPartial,selector,value,indices)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartial" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam> 
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, bool selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            return ValueAverageConditional(enterPartial, selector, indices, result);
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enterPartial">Incoming message from <c>enterPartial</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>value</c>. Because the factor is deterministic, <c>enterPartial</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(selector) p(selector) log(sum_enterPartial p(enterPartial) factor(enterPartial,selector,value,indices)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartial" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam> 
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, Bernoulli selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                double scale = (indices[0] == 0) ? selector.GetProbTrue() : selector.GetProbFalse();
                result.SetToPower(enterPartial[0], scale);
                if (indices.Length > 1)
                {
                    // TODO: use pre-allocated buffer
                    TDist power = (TDist)result.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        scale = (indices[i] == 0) ? selector.GetProbTrue() : selector.GetProbFalse();
                        power.SetToPower(enterPartial[i], scale);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }

        /// <summary>VMP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Discrete SelectorAverageLogarithm(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Bernoulli SelectorAverageLogarithm(Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

#if false
    /// <summary>
    /// VMP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except 'cases'.
    /// Because the factor is deterministic, 'enterPartial' is integrated out before taking the logarithm.
    /// The formula is <c>exp(sum_(value) p(value) log(sum_enterPartial p(enterPartial) factor(enterPartial,cases,value,indices)))</c>.
    /// </para></remarks>
		[Skip]
		public static BernoulliList CasesAverageLogarithm<BernoulliList>(BernoulliList result)
			where BernoulliList : SettableToUniform
		{
			result.SetToUniform();
			return result;
		}
		/// <summary>
		/// VMP message to 'value'
		/// </summary>
		/// <param name="enterPartial">Incoming message from 'enterPartial'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="cases">Incoming message from 'cases'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
		/// <param name="indices">Constant value for 'indices'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except 'value'.
		/// Because the factor is deterministic, 'enterPartial' is integrated out before taking the logarithm.
		/// The formula is <c>exp(sum_(cases) p(cases) log(sum_enterPartial p(enterPartial) factor(enterPartial,cases,value,indices)))</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enterPartial"/> is not a proper distribution</exception>
		/// <exception cref="ImproperMessageException"><paramref name="cases"/> is not a proper distribution</exception>
		public static TDist ValueAverageLogarithm<TDist>([SkipIfAllUniform] IList<TDist> enterPartial, [SkipIfUniform] IList<Bernoulli> cases, int[] indices, TDist result)
			where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
		{
			if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
			if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
			if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
			else {
				double scale = Math.Exp(cases[indices[0]].LogOdds);
				result.SetToPower(enterPartial[0], scale);
				if (indices.Length > 1) {
					// TODO: use pre-allocated buffer
					TDist power = (TDist)result.Clone();
					for (int i = 1; i < indices.Length; i++) {
						scale = Math.Exp(cases[indices[i]].LogOdds);
						power.SetToPower(enterPartial[i], scale);
						result.SetToProduct(result, power);
					}
				}
			}
			return result;
		}
#endif
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.EnterPartialTwo{T}(bool, bool, T, int[])" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Gate), "EnterPartialTwo<>")]
    [Quality(QualityBand.Mature)]
    public static class GateEnterPartialTwoOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterPartialTwo,case0,case1,value,indices))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterPartialTwo,case0,case1,value,indices))</c>.</para>
        /// </remarks>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>enterPartialTwo</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enterPartialTwo</c> as the random arguments are varied. The formula is <c>proj[p(enterPartialTwo) sum_(value) p(value) factor(enterPartialTwo,case0,case1,value,indices)]/p(enterPartialTwo)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="value" /> is not a proper distribution.</exception>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialTwoAverageConditional<TValue, TResultList>([SkipIfUniform] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <summary>EP message to <c>case0</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>case0</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Bernoulli Case0AverageConditional(Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>EP message to <c>case1</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>case1</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Bernoulli Case1AverageConditional(Bernoulli result)
        {
            return Case0AverageConditional(result);
        }

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
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue ValueAverageConditional<TValue>(
            [SkipIfAllUniform] IList<TValue> enterPartialTwo, Bernoulli case0, Bernoulli case1, TValue value, int[] indices, TValue result)
            where TValue : ICloneable, SettableToUniform, SettableToProduct<TValue>, SettableToRatio<TValue>, SettableToWeightedSum<TValue>, CanGetLogAverageOf<TValue>
        {
            if (indices.Length != enterPartialTwo.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartialTwo.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                // TODO: use pre-allocated buffers
                result.SetToProduct(value, enterPartialTwo[0]);
                double scale = Math.Exp((indices[0] == 0 ? case0 : case1).LogOdds);
                double sumCases = scale;
                double resultScale = scale;
                if (indices.Length > 1)
                {
                    TValue product = (TValue)value.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        product.SetToProduct(value, enterPartialTwo[i]);
                        scale = Math.Exp((indices[i] == 0 ? case0 : case1).LogOdds);
                        result.SetToSum(resultScale, result, scale, product);
                        resultScale += scale;
                        sumCases += scale;
                    }
                }
                double totalCases = Math.Exp(case0.LogOdds) + Math.Exp(case1.LogOdds);
                result.SetToSum(resultScale, result, totalCases - sumCases, value);
                result.SetToRatio(result, value, GateEnterOp<TValue>.ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterPartialTwo">Incoming message from <c>enterPartialTwo</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="case1">Constant value for <c>case1</c>.</param>
        /// <param name="case2" />
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterPartialTwo) p(enterPartialTwo) factor(enterPartialTwo,case0,case1,value,indices)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartialTwo" /> is not a proper distribution.</exception>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TDomain">The type of the variable entering the gate.</typeparam>
        public static TValue ValueAverageConditional<TValue, TDomain>(
            [SkipIfAllUniform] IList<TValue> enterPartialTwo, bool case1, bool case2, int[] indices, TValue result)
            where TValue : IDistribution<TDomain>, SettableTo<TValue>
        {
            if (indices.Length != enterPartialTwo.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartialTwo.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if ((indices[i] == 0 && case1) || (indices[i] == 1 && case2))
                    {
                        result.SetTo(enterPartialTwo[indices[i]]);
                        break;
                    }
                }
                return result;
            }
        }

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterPartialTwo,case0,case1,value,indices))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>enterPartialTwo</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enterPartialTwo</c> as the random arguments are varied. The formula is <c>proj[sum_(value) p(value) factor(enterPartialTwo,case0,case1,value,indices)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="value" /> is not a proper distribution.</exception>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialTwoAverageLogarithm<TValue, TResultList>([SkipIfUniform] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <summary>VMP message to <c>case0</c>.</summary>
        /// <param name="enterPartialTwo">Incoming message from <c>enterPartialTwo</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>case0</c>. Because the factor is deterministic, <c>enterPartialTwo</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(value) p(value) log(sum_enterPartialTwo p(enterPartialTwo) factor(enterPartialTwo,case0,case1,value,indices)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        [Skip]
        public static Bernoulli Case0AverageLogarithm<TValue>(IList<TValue> enterPartialTwo, TValue value, int[] indices, Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>case1</c>.</summary>
        /// <param name="enterPartialTwo">Incoming message from <c>enterPartialTwo</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>case1</c>. Because the factor is deterministic, <c>enterPartialTwo</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(value) p(value) log(sum_enterPartialTwo p(enterPartialTwo) factor(enterPartialTwo,case0,case1,value,indices)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        [Skip]
        public static Bernoulli Case1AverageLogarithm<TValue>(IList<TValue> enterPartialTwo, TValue value, int[] indices, Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enterPartialTwo">Incoming message from <c>enterPartialTwo</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>value</c>. Because the factor is deterministic, <c>enterPartialTwo</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(case0,case1) p(case0,case1) log(sum_enterPartialTwo p(enterPartialTwo) factor(enterPartialTwo,case0,case1,value,indices)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterPartialTwo" /> is not a proper distribution.</exception>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue ValueAverageLogarithm<TValue>(
            [SkipIfAllUniform] IList<TValue> enterPartialTwo, Bernoulli case0, Bernoulli case1, int[] indices, TValue result)
            where TValue : ICloneable, SettableToProduct<TValue>, SettableToPower<TValue>
        {
            if (indices.Length != enterPartialTwo.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartialTwo.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                double scale = Math.Exp((indices[0] == 0 ? case0 : case1).LogOdds);
                result.SetToPower(enterPartialTwo[0], scale);
                if (indices.Length > 1)
                {
                    // TODO: use pre-allocated buffer
                    TValue power = (TValue)result.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        scale = Math.Exp((indices[i] == 0 ? case0 : case1).LogOdds);
                        power.SetToPower(enterPartialTwo[i], scale);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.EnterOne{T}(int, T, int)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable entering the gate.</typeparam>
    [FactorMethod(typeof(Gate), "EnterOne<>", null, typeof(int), null, typeof(int))]
    [Quality(QualityBand.Mature)]
    public static class GateEnterOneOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterOne,selector,value,index))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterOne,selector,value,index))</c>.</para>
        /// </remarks>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>enterOne</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <returns>The outgoing EP message to the <c>enterOne</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enterOne</c> as the random arguments are varied. The formula is <c>proj[p(enterOne) sum_(value) p(value) factor(enterOne,selector,value,index)]/p(enterOne)</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        public static TValue EnterOneAverageConditional<TValue>([IsReturned] TValue value)
        {
            return value;
        }

        /// <summary>EP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Discrete SelectorAverageConditional(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

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
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>, SettableTo<TDist>
        {
            double logProb = selector.GetLogProb(index);
            if (logProb == 0.0)
            {
                result.SetTo(enterOne);
            }
            else if (double.IsNegativeInfinity(logProb))
            {
                result.SetToUniform();
            }
            else
            {
                result.SetToProduct(value, enterOne);
                double logOtherProb = MMath.Log1MinusExp(logProb);
                double shift = Math.Max(logProb, logOtherProb);
                // avoid (-Infinity) - (-Infinity)
                if (Double.IsNegativeInfinity(shift))
                    throw new AllZeroException();
                result.SetToSum(Math.Exp(logProb - shift), result, Math.Exp(logOtherProb - shift), value);
                result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enterOne">Incoming message from <c>enterOne</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="selector">Constant value for <c>selector</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enterOne) p(enterOne) factor(enterOne,selector,value,index)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterOne" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, int selector, int index, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (selector == index)
                result.SetTo(enterOne);
            else
                result.SetToUniform();
            return result;
        }

#if false
    /// <summary>
    /// EP message to 'b'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'b' conditioned on the given values.
    /// </para></remarks>
		[Skip]
		public static Bernoulli BAverageConditional(Bernoulli result)
		{
			result.SetToUniform();
			return result;
		}
		/// <summary>
		/// EP message to 'value'
		/// </summary>
		/// <param name="enterOne">Incoming message from 'enterOne'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="cases">Incoming message from 'cases'.</param>
		/// <param name="value">Incoming message from 'value'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="index">Constant value for 'index'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
		/// The formula is <c>proj[p(value) sum_(enterOne,cases) p(enterOne,cases) factor(enterOne,cases,value,index)]/p(value)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enterOne"/> is not a proper distribution</exception>
		/// <exception cref="ImproperMessageException"><paramref name="value"/> is not a proper distribution</exception>
		public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, Bernoulli b, [Proper] TDist value, TDist result)
			where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>, SettableTo<TDist>
		{
			double logProb = b.LogOdds;
			if (logProb == 0.0) {
				result.SetTo(enterOne);
			} else if (double.IsNegativeInfinity(logProb)) {
				result.SetToUniform();
			} else {
				result.SetToProduct(value, enterOne);
				double logOtherProb = MMath.Log1MinusExp(logProb);
				double shift = Math.Max(logProb, logOtherProb);
				// avoid (-Infinity) - (-Infinity)
				if (Double.IsNegativeInfinity(shift)) throw new AllZeroException();
				result.SetToSum(Math.Exp(logProb - shift), result, Math.Exp(logOtherProb - shift), value);
				if (GateEnterOp<T>.ForceProper && (result is Gaussian)) {
					Gaussian r = (Gaussian)(object)result;
					r.SetToRatioProper(r, (Gaussian)(object)value);
					result = (TDist)(object)r;
				} else {
					result.SetToRatio(result, value);
				}
			}
			return result;
		}
		/// <summary>
		/// EP message to 'value'
		/// </summary>
		/// <param name="enterOne">Incoming message from 'enterOne'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="cases">Incoming message from 'cases'.</param>
		/// <param name="index">Constant value for 'index'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
		/// The formula is <c>proj[p(value) sum_(enterOne,cases) p(enterOne,cases) factor(enterOne,cases,value,index)]/p(value)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enterOne"/> is not a proper distribution</exception>
		public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, bool b, TDist result)
			where TDist : IDistribution<T>, SettableTo<TDist>
		{
			if (b)
				result.SetTo(enterOne);
			else
				result.SetToUniform();
			return result;
		}
#endif

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enterOne,selector,value,index))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>enterOne</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <returns>The outgoing VMP message to the <c>enterOne</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enterOne</c> as the random arguments are varied. The formula is <c>proj[sum_(value) p(value) factor(enterOne,selector,value,index)]</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        public static TValue EnterOneAverageLogarithm<TValue>([IsReturned] TValue value)
        {
            return value;
        }

        /// <summary>VMP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Discrete SelectorAverageLogarithm(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enterOne">Incoming message from <c>enterOne</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>value</c>. Because the factor is deterministic, <c>enterOne</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(selector) p(selector) log(sum_enterOne p(enterOne) factor(enterOne,selector,value,index)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enterOne" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>([SkipIfUniform] TDist enterOne, Discrete selector, int index, TDist result)
            where TDist : SettableToPower<TDist>
        {
            double scale = selector[index];
            result.SetToPower(enterOne, scale);
            return result;
        }

#if false
    /// <summary>
    /// VMP message to 'b'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'b' conditioned on the given values.
    /// </para></remarks>
		[Skip]
		public static Bernoulli BAverageLogarithm(Bernoulli result)
		{
			result.SetToUniform();
			return result;
		}
		/// <summary>
		/// VMP message to 'value'
		/// </summary>
		/// <param name="enterOne">Incoming message from 'enterOne'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="b">Incoming message from 'b'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		public static TDist ValueAverageLogarithm<TDist>([SkipIfUniform] TDist enterOne, Bernoulli b, TDist result)
			where TDist : SettableToPower<TDist>
		{
			double scale = Math.Exp(b.LogOdds);
			result.SetToPower(enterOne, scale);
			return result;
		}
#endif
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Gate.Enter{T}(int, T)" /></description></item><item><description><see cref="Gate.Enter{T}(bool, T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(bool), null)]
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(int), null)]
    [Quality(QualityBand.Mature)]
    public static class GateEnterOp<T>
    {
        /// <summary>
        /// Force proper messages
        /// </summary>
        public static bool ForceProper = true;

        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enter,selector,value))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enter,selector,value))</c>.</para>
        /// </remarks>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>enter</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enter</c> as the random arguments are varied. The formula is <c>proj[p(enter) sum_(value) p(value) factor(enter,selector,value)]/p(enter)</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterAverageConditional<TValue, TResultList>([IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <summary>Initialize the buffer <c>enter</c>.</summary>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="factory" />
        /// <returns>Initial value of buffer <c>enter</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        /// <typeparam name="TArray">The type of an array that can be produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static TArray EnterInit<TValue, TArray>(
            Discrete selector, [IgnoreDependency] TValue value, IArrayFactory<TValue, TArray> factory)
            where TValue : ICloneable
        {
            return factory.CreateArray(selector.Dimension, i => (TValue)value.Clone());
        }

        /// <summary>EP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Discrete SelectorAverageConditional(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

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
            where TDist : IDistribution<T>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (selector.Dimension != enter.Count)
                throw new ArgumentException("selector.Dimension != enter.Count");
            // TODO: use pre-allocated buffers
            double logProbSum = selector.GetLogProb(0);
            if (!double.IsNegativeInfinity(logProbSum))
            {
                result.SetToProduct(value, enter[0]);
            }
            if (selector.Dimension > 1)
            {
                TDist product = (TDist)value.Clone();
                for (int i = 1; i < selector.Dimension; i++)
                {
                    double logProb = selector.GetLogProb(i);
                    double shift = Math.Max(logProbSum, logProb);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == selector.Dimension - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        double productWeight = Math.Exp(logProb - shift);
                        if (productWeight > 0)
                        {
                            product.SetToProduct(value, enter[i]);
                            result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                            logProbSum = MMath.LogSumExp(logProbSum, logProb);
                        }
                    }
                }
            }
            result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            return result;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enter">Incoming message from <c>enter</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Constant value for <c>selector</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enter) p(enter) factor(enter,selector,value)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enter" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, int selector, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            result.SetTo(enter[selector]);
            return result;
        }

#if false
    /// <summary>
    /// EP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'cases' conditioned on the given values.
    /// </para></remarks>
		[Skip]
		public static BernoulliList CasesAverageConditional<BernoulliList>(BernoulliList result)
			where BernoulliList : SettableToUniform
		{
			result.SetToUniform();
			return result;
		}
		/// <summary>
		/// EP message to 'value'
		/// </summary>
		/// <param name="enter">Incoming message from 'enter'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="cases">Incoming message from 'cases'.</param>
		/// <param name="value">Incoming message from 'value'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
		/// The formula is <c>proj[p(value) sum_(enter,cases) p(enter,cases) factor(enter,cases,value)]/p(value)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enter"/> is not a proper distribution</exception>
		public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, IList<Bernoulli> cases, TDist value, TDist result)
			where TDist : IDistribution<T>, SettableToProduct<TDist>,
			SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
		{
			if (cases.Count < enter.Count) throw new ArgumentException("cases.Count < enter.Count");
			// TODO: use pre-allocated buffers
			double logProbSum = cases[0].LogOdds;
			if (!double.IsNegativeInfinity(logProbSum)) {
				result.SetToProduct(value, enter[0]);
			}
			if (cases.Count > 1) {
				TDist product = (TDist)value.Clone();
				for (int i = 1; i < cases.Count; i++) {
					double logProb = cases[i].LogOdds;
					double shift = Math.Max(logProbSum, logProb);
					// avoid (-Infinity) - (-Infinity)
					if (Double.IsNegativeInfinity(shift)) {
						if (i == cases.Count - 1) {
							throw new AllZeroException();
						}
						// do nothing
					} else {
						double productWeight = Math.Exp(logProb - shift);
						if (productWeight > 0) {
							product.SetToProduct(value, enter[i]);
							result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
							logProbSum = MMath.LogSumExp(logProbSum, logProb);
						}
					}
				}
			}
			if (ForceProper && (result is Gaussian)) {
				Gaussian r = (Gaussian)(object)result;
				r.SetToRatioProper(r, (Gaussian)(object)value);
				result = (TDist)(object)r;
			} else {
				result.SetToRatio(result, value);
			}
			return result;
		}
		/// <summary>
		/// EP message to 'value'
		/// </summary>
		/// <param name="enter">Incoming message from 'enter'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="cases">Constant value for 'cases'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
		/// The formula is <c>proj[p(value) sum_(enter) p(enter) factor(enter,cases,value)]/p(value)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enter"/> is not a proper distribution</exception>
		public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, bool[] cases, TDist result)
			where TDist : IDistribution<T>, SettableTo<TDist>
		{
			if (cases.Length < enter.Count) throw new ArgumentException("cases.Count < enter.Count");
			result.SetToUniform();
			for (int i = 0; i < cases.Length; i++) {
				if (cases[i]) {
					result.SetTo(enter[i]);
					break;
				}
			}
			return result;
		}
#endif

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enter,selector,value))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>enter</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enter</c> as the random arguments are varied. The formula is <c>proj[sum_(value) p(value) factor(enter,selector,value)]</c>.</para>
        /// </remarks>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterAverageLogarithm<TValue, TResultList>([IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <summary>VMP message to <c>selector</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>selector</c> conditioned on the given values.</para>
        /// </remarks>
        [Skip]
        public static Discrete SelectorAverageLogarithm(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enter">Incoming message from <c>enter</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>value</c>. Because the factor is deterministic, <c>enter</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(selector) p(selector) log(sum_enter p(enter) factor(enter,selector,value)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enter" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>([SkipIfAllUniform] IList<TDist> enter, Discrete selector, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (selector.Dimension != enter.Count)
                throw new ArgumentException("selector.Dimension != enterPartial.Count");
            double scale = selector[0];
            result.SetToPower(enter[0], scale);
            if (selector.Dimension > 1)
            {
                // TODO: use pre-allocated buffer
                TDist power = (TDist)result.Clone();
                for (int i = 1; i < selector.Dimension; i++)
                {
                    scale = selector[i];
                    power.SetToPower(enter[i], scale);
                    result.SetToProduct(result, power);
                }
            }
            return result;
        }

#if false
    /// <summary>
    /// VMP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'cases' conditioned on the given values.
    /// </para></remarks>
		[Skip]
		public static BernoulliList CasesAverageLogarithm<BernoulliList>(BernoulliList result)
			where BernoulliList : SettableToUniform
		{
			result.SetToUniform();
			return result;
		}
		// result = prod_i enterPartial[i]^cases[indices[i]]
		/// <summary>
		/// VMP message to 'value'
		/// </summary>
		/// <param name="enter">Incoming message from 'enter'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="cases">Incoming message from 'cases'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except 'value'.
		/// Because the factor is deterministic, 'enter' is integrated out before taking the logarithm.
		/// The formula is <c>exp(sum_(cases) p(cases) log(sum_enter p(enter) factor(enter,cases,value)))</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="enter"/> is not a proper distribution</exception>
		public static TDist ValueAverageLogarithm<TDist>([SkipIfAllUniform] IList<TDist> enter, IList<Bernoulli> cases, TDist result)
			where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
		{
			if (cases.Count < enter.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
			double scale = Math.Exp(cases[0].LogOdds);
			result.SetToPower(enter[0], scale);
			if (cases.Count > 1) {
				// TODO: use pre-allocated buffer
				TDist power = (TDist)result.Clone();
				for (int i = 1; i < cases.Count; i++) {
					scale = Math.Exp(cases[i].LogOdds);
					power.SetToPower(enter[i], scale);
					result.SetToProduct(result, power);
				}
			}
			return result;
		}
#endif
    }
}
