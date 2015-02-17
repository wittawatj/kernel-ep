// (C) Copyright 2008 Microsoft Research Cambridge

#define UseRatioDir

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

#if true
    /// <summary>Provides outgoing messages for <see cref="Gate.ExitingVariable{T}(T, out T)" />, given random arguments to the function.</summary>
    /// <remarks><para>
    /// This factor is like <see cref="Factor.ReplicateWithMarginal{T}"/> except <c>Uses[0]</c> plays the role of <c>Def</c>,
    /// and <c>Def</c> is considered a <c>Use</c>. Needed only when a variable exits a gate in VMP.
    /// </para></remarks>
    [FactorMethod(typeof(Gate), "ExitingVariable<>")]
    [Quality(QualityBand.Mature)]
    public static class ExitingVariableOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Use,Def,Marginal))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>Marginal</c>.</summary>
        /// <param name="Use">Incoming message from <c>Use</c>.</param>
        /// <returns>The outgoing VMP message to the <c>Marginal</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Marginal</c> with <c>Use</c> integrated out. The formula is <c>sum_Use p(Use) factor(Use,Def,Marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T MarginalAverageLogarithm<T>([IsReturned] T Use)
        {
            return Use;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip]
        public static T MarginalAverageLogarithmInit<T>(T Def)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <summary>VMP message to <c>Use</c>.</summary>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <returns>The outgoing VMP message to the <c>Use</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Use</c> as the random arguments are varied. The formula is <c>proj[sum_(Def) p(Def) factor(Use,Def,Marginal)]</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T UseAverageLogarithm<T>([IsReturned] T Def)
        {
            return Def;
        }

        /// <summary>VMP message to <c>Def</c>.</summary>
        /// <param name="Use">Incoming message from <c>Use</c>.</param>
        /// <returns>The outgoing VMP message to the <c>Def</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Def</c> with <c>Use</c> integrated out. The formula is <c>sum_Use p(Use) factor(Use,Def,Marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageLogarithm<T>([IsReturned] T Use)
        {
            return Use;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.ReplicateExiting{T}(T, int)" />, given random arguments to the function.</summary>
    /// <remarks><para>
    /// This factor is like <see cref="Factor.Replicate{T}"/> except <c>Uses[0]</c> plays the role of <c>Def</c>,
    /// and <c>Def</c> is considered a <c>Use</c>. Needed only when a variable exits a gate in VMP.
    /// </para></remarks>
    [FactorMethod(typeof(Gate), "ReplicateExiting<>")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateExitingOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Uses,Def,count))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[sum_(Def) p(Def) factor(Uses,Def,count)]</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T UsesAverageLogarithm<T>([AllExceptIndex] IList<T> Uses, T Def, int resultIndex, T result)
            where T : SettableTo<T>, SettableToProduct<T>
        {
            if (resultIndex == 0)
            {
                result.SetTo(Def);
                result = Distribution.SetToProductWithAllExcept(result, Uses, 0);
            }
            else
            {
                result.SetTo(Uses[0]);
            }
            return result;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip]
        public static T UsesAverageLogarithmInit<T>(T Def, int resultIndex)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <summary>VMP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Def</c> with <c>Uses</c> integrated out. The formula is <c>sum_Uses p(Uses) factor(Uses,Def,count)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageLogarithm<T>([SkipIfAllUniform] IList<T> Uses, T result)
            where T : SettableTo<T>
        {
            result.SetTo(Uses[0]);
            return result;
        }
    }
#else
    /// <summary>
    /// Provides outgoing messages for <see cref="Gate.ExitingVariable{T}"/>, given random arguments to the function.
    /// </summary>
    /// <remarks><para>
    /// This factor is like ReplicateWithMarginal except Uses[0] plays the role of Def, and Def is
    /// considered a Use.  Needed only when a variable exits a gate in VMP.
    /// </para></remarks>
	[FactorMethod(typeof(Gate), "ExitingVariable<>")]
	[Quality(QualityBand.Preview)]
	public static class ExitingVariableOp
	{
		/// <summary>
		/// Evidence message for VMP.
		/// </summary>
		/// <returns><c>sum_x marginal(x)*log(factor(x))</c></returns>
		/// <remarks><para>
		/// The formula for the result is <c>int log(f(x)) q(x) dx</c>
		/// where <c>x = (Uses,Def,Marginal)</c>.
		/// </para></remarks>
		[Skip]
		public static double AverageLogFactor() { return 0.0; }

		/// <summary>
		/// VMP message to 'Marginal'.
		/// </summary>
		/// <param name="Uses">Incoming message from 'Uses'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="result">Modified to contain the outgoing message.</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'Marginal'.
		/// The formula is <c>int log(f(Marginal,x)) q(x) dx</c> where <c>x = (Uses,Def)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="Uses"/> is not a proper distribution</exception>
		public static T MarginalAverageLogarithm<T>([SkipIfAllUniform] IList<T> Uses, T result)
		where T : SettableTo<T>
		{
			result.SetTo(Uses[0]);
			return result;
		}

		/// <summary>
		/// VMP message to 'Uses'.
		/// </summary>
		/// <param name="Uses">Incoming message from 'Uses'.</param>
		/// <param name="Def">Incoming message from 'Def'.</param>
		/// <param name="resultIndex">Index of the 'Uses' array for which a message is desired.</param>
		/// <param name="result">Modified to contain the outgoing message.</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'Uses'.
		/// The formula is <c>int log(f(Uses,x)) q(x) dx</c> where <c>x = (Def,Marginal)</c>.
		/// </para></remarks>
		[SkipIfAllUniform]
		public static T UsesAverageLogarithm<T>([AllExceptIndex] IList<T> Uses, T Def, int resultIndex, T result)
				where T : SettableTo<T>, SettableToProduct<T>
		{
			if (resultIndex == 0) {
				result.SetTo(Def);
				result = Distribution.SetToProductWithAllExcept(result, Uses, 0);
			} else {
				result.SetTo(Uses[0]);
			}
			return result;
		}

		/// <summary>
		/// VMP message to 'Def'.
		/// </summary>
		/// <param name="Uses">Incoming message from 'Uses'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
		/// <param name="result">Modified to contain the outgoing message.</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'Def'.
		/// The formula is <c>int log(f(Def,x)) q(x) dx</c> where <c>x = (Uses,Marginal)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="Uses"/> is not a proper distribution</exception>
		public static T DefAverageLogarithm<T>([SkipIfAllUniform] IList<T> Uses, T result)
		where T : SettableTo<T>
		{
			return MarginalAverageLogarithm(Uses, result);
		}
	}
#endif

    /// <summary>Provides outgoing messages for <see cref="Gate.Exit{T}(bool[], T[])" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable exiting the gate.</typeparam>
    [FactorMethod(typeof(Gate), "Exit<>")]
    [Quality(QualityBand.Mature)]
    public static class GateExitOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exit,cases) p(exit,cases) factor(exit,cases,values) / sum_exit p(exit) messageTo(exit))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<TDist>(TDist exit, IList<bool> cases)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <param name="to_exit">Outgoing message to <c>exit</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exit,cases,values) p(exit,cases,values) factor(exit,cases,values) / sum_exit p(exit) messageTo(exit))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exit" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static double LogEvidenceRatio<TDist>([SkipIfUniform] TDist exit, IList<Bernoulli> cases, IList<TDist> values, [Fresh] TDist to_exit)
            where TDist : IDistribution<T>, CanGetLogAverageOf<TDist>
        {
            return -to_exit.GetLogAverageOf(exit);
        }

        /// <summary>EP message to <c>values</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>values</c> as the random arguments are varied. The formula is <c>proj[p(values) sum_(exit) p(exit) factor(exit,cases,values)]/p(values)</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the message from <c>exit</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageConditional<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            result.SetAllElementsTo(exit);
            return result;
        }

        /// <summary>EP message to <c>cases</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>cases</c> as the random arguments are varied. The formula is <c>proj[p(cases) sum_(exit,values) p(exit,values) factor(exit,cases,values)]/p(cases)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exit" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TBernoulliList">The type of the outgoing message.</typeparam>
        public static TBernoulliList CasesAverageConditional<TDist, TBernoulliList>(
            [SkipIfUniform] TDist exit, IList<TDist> values, TBernoulliList result)
            where TBernoulliList : IList<Bernoulli>
            where TDist : IDistribution<T>, CanGetLogAverageOf<TDist>
        {
            for (int i = 0; i < values.Count; i++)
            {
                result[i] = Bernoulli.FromLogOdds(exit.GetLogAverageOf(values[i]));
            }
            return result;
        }

        /// <summary>EP message to <c>exit</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
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
        public static TDist ExitAverageConditional<TDist>(TDist exit, IList<bool> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>
        {
            for (int i = 0; i < cases.Count; i++)
                if (cases[i])
                {
                    result.SetTo(values[i]);
                    return result;
                }

            throw new ApplicationException("no case is true");
        }

        /// <summary>EP message to <c>exit</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
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
        public static TDist ExitAverageConditional<TDist>(TDist exit, IList<Bernoulli> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                double resultScale = exit.GetLogAverageOf(values[0]) + cases[0].LogOdds;
                if (double.IsNaN(resultScale))
                    throw new AllZeroException();
                int resultIndex = 0;
                // TODO: use pre-allocated buffer
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    double scale = exit.GetLogAverageOf(values[i]) + cases[i].LogOdds;
                    double shift = Math.Max(resultScale, scale);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        double weight1 = Math.Exp(resultScale - shift);
                        double weight2 = Math.Exp(scale - shift);
                        if (weight2 > 0)
                        {
                            if (weight1 == 0)
                            {
                                resultIndex = i;
                                resultScale = scale;
                            }
                            else
                            {
                                if (resultIndex >= 0)
                                {
                                    result.SetToProduct(exit, values[resultIndex]);
                                    resultIndex = -1;
                                }
                                product.SetToProduct(exit, values[i]);
                                result.SetToSum(weight1, result, weight2, product);
                                resultScale = MMath.LogSumExp(resultScale, scale);
                            }
                        }
                    }
                }
                if (resultIndex >= 0)
                {
                    // result is simply values[resultIndex]
                    return values[resultIndex];
                }
                result.SetToRatio(result, exit, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>exit</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
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
        public static TDist ExitAverageConditional1<TDist>(TDist exit, IList<Bernoulli> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                double resultScale = Math.Exp(exit.GetLogAverageOf(values[0]) + cases[0].LogOdds);
                if (double.IsNaN(resultScale))
                    throw new AllZeroException();
                if (resultScale > 0)
                {
                    result.SetToProduct(exit, values[0]);
                }
                // TODO: use pre-allocated buffer
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    double scale = Math.Exp(exit.GetLogAverageOf(values[i]) + cases[i].LogOdds);
                    if (scale > 0)
                    {
                        product.SetToProduct(exit, values[i]);
                        result.SetToSum(resultScale, result, scale, product);
                        resultScale += scale;
                    }
                }
                result.SetToRatio(result, exit, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>exit</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
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
        public static TDist ExitAverageConditional2<TDist>(
            TDist exit, IList<Bernoulli> cases, [SkipIfAllUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>, ICloneable, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                result.SetToProduct(exit, values[0]);
                double scale = cases[0].LogOdds;
                double resultScale = scale;
                // TODO: use pre-allocated buffer
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    scale = cases[i].LogOdds;
                    double shift = Math.Max(resultScale, scale);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        double weight1 = Math.Exp(resultScale - shift);
                        double weight2 = Math.Exp(scale - shift);
                        if (weight2 > 0)
                        {
                            product.SetToProduct(exit, values[i]);
                            result.SetToSum(weight1, result, weight2, product);
                            resultScale = MMath.LogSumExp(resultScale, scale);
                        }
                    }
                }
                result.SetToRatio(result, exit, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>cases</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>cases</c> as the random arguments are varied. The formula is <c>proj[p(cases) sum_(exit,values) p(exit,values) factor(exit,cases,values)]/p(cases)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TBernoulliList">The type of the outgoing message.</typeparam>
        public static TBernoulliList CasesAverageConditional<TDist, TBernoulliList>(TDist exit, IList<T> values, TBernoulliList result)
            where TBernoulliList : IList<Bernoulli>
            where TDist : CanGetLogProb<T>
        {
            for (int i = 0; i < values.Count; i++)
            {
                result[i] = Bernoulli.FromLogOdds(exit.GetLogProb(values[i]));
            }
            return result;
        }

        /// <summary>EP message to <c>exit</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exit</c> as the random arguments are varied. The formula is <c>proj[p(exit) sum_(cases,values) p(cases,values) factor(exit,cases,values)]/p(exit)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional<TDist>(
            TDist exit, IList<Bernoulli> cases, IList<T> values, TDist result)
            where TDist : ICloneable, HasPoint<T>, SettableToWeightedSum<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.Point = values[0];
            }
            else
            {
                result.Point = values[0];
                double scale = cases[0].LogOdds;
                double resultScale = scale;
                // TODO: overload SetToSum to accept constants.
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    product.Point = values[i];
                    scale = cases[i].LogOdds;
                    double shift = Math.Max(resultScale, scale);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        result.SetToSum(Math.Exp(resultScale - shift), result, Math.Exp(scale - shift), product);
                        resultScale = MMath.LogSumExp(resultScale, scale);
                    }
                }
            }
            return result;
        }

        /// <summary />
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static TDist ExitAverageConditionalInit<TDist>([IgnoreDependency] IList<TDist> values)
            where TDist : ICloneable
        {
            return (TDist)values[0].Clone();
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_exit">Outgoing message to <c>exit</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exit" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static double AverageLogFactor<TDist>([SkipIfUniform] TDist exit, [Fresh] TDist to_exit)
            where TDist : IDistribution<T>, CanGetAverageLog<TDist>
        {
            // cancel the evidence message from the child variable's child factors
            return -to_exit.GetAverageLog(exit);
        }

        /// <summary>VMP message to <c>values</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>values</c> with <c>exit</c> integrated out. The formula is <c>sum_exit p(exit) factor(exit,cases,values)</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the message from <c>exit</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageLogarithm<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            result.SetAllElementsTo(exit);
            return result;
        }

#if true

        /// <summary>VMP message to <c>cases</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>cases</c>. Because the factor is deterministic, <c>exit</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(values) p(values) log(sum_exit p(exit) factor(exit,cases,values)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exit" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TBernoulliList">The type of the outgoing message.</typeparam>
        [NoTriggers] // see VmpTests.GateExitTriggerTest
        public static TBernoulliList CasesAverageLogarithm<TDist, TBernoulliList>(
            [SkipIfUniform] TDist exit, [SkipIfAllUniform, Proper, Trigger] IList<TDist> values, TBernoulliList result)
            where TBernoulliList : IList<Bernoulli>
            where TDist : CanGetAverageLog<TDist>
        {
            for (int i = 0; i < values.Count; i++)
            {
                result[i] = Bernoulli.FromLogOdds(values[i].GetAverageLog(exit));
            }
            return result;
        }

        /// <summary>VMP message to <c>exit</c>.</summary>
        /// <param name="exit">Incoming message from <c>exit</c>.</param>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exit</c> as the random arguments are varied. The formula is <c>proj[sum_(cases,values) p(cases,values) factor(exit,cases,values)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageLogarithm<TDist>(TDist exit, IList<bool> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>
        {
            return ExitAverageConditional(exit, cases, values, result);
        }

        /// <summary>VMP message to <c>exit</c>.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exit</c> as the random arguments are varied. The formula is <c>proj[sum_(cases,values) p(cases,values) factor(exit,cases,values)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageLogarithm<TDist>(IList<Bernoulli> cases, [SkipIfAllUniform, Proper] IList<TDist> values, TDist result)
            where TDist : ICloneable, SettableToProduct<TDist>,
                SettableToPower<TDist>, CanGetAverageLog<TDist>,
                SettableToUniform, SettableTo<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>
        {
            // result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
#if DEBUG
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
#endif
            TDist uniform = (TDist)result.Clone();
            uniform.SetToUniform();
            return ExitAverageConditional2<TDist>(uniform, cases, values, result);
        }

        /// <summary />
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static TDist ExitAverageLogarithmInit<TDist>([IgnoreDependency] IList<TDist> values)
            where TDist : ICloneable
        {
            return (TDist)values[0].Clone();
        }
#else
		[Skip]
		public static DistributionArray<Bernoulli> CasesAverageLogarithm(DistributionArray<Bernoulli> result)
		{
			return result;
		}
		// result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
		public static T ExitAverageLogarithm<T>(DistributionArray<Bernoulli> cases, [SkipIfUniform] DistributionArray<T> values, T result)
			where T : Diffable, SettableTo<T>, ICloneable, SettableToUniform, SettableToProduct<T>,
			SettableToPower<T>, SettableToRatio<T>, SettableToWeightedSum<T>, LogInnerProductable<T>, CanGetAverageLog<T>
		{
			if (cases.Count != values.Count) throw new ArgumentException("cases.Count != values.Count");
			if (cases.Count == 0) throw new ArgumentException("cases.Count == 0");
			else {
				result.SetToPower(values[0], cases[0].LogOdds);
				if (cases.Count > 1) {
					// TODO: use pre-allocated buffer
					T power = (T)result.Clone();
					for (int i = 1; i < cases.Count; i++) {
						power.SetToPower(values[i], cases[i].LogOdds);
						result.SetToProduct(result, power);
					}
				}
			}
			return result;
		}
#endif
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.ExitTwo{T}(bool, bool, T[])" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Gate), "ExitTwo<>")]
    [Quality(QualityBand.Mature)]
    public static class GateExitTwoOp
    {
        /// <summary>EP message to <c>values</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>values</c> as the random arguments are varied. The formula is <c>proj[p(values) sum_(exitTwo) p(exitTwo) factor(exitTwo,case0,case1,values)]/p(values)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exitTwo" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageConditional<TExit, TResultList>([SkipIfUniform] TExit exitTwo, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            result.SetAllElementsTo(exitTwo);
            return result;
        }

        /// <summary>EP message to <c>case0</c>.</summary>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>case0</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>case0</c> as the random arguments are varied. The formula is <c>proj[p(case0) sum_(values) p(values) factor(exitTwo,case0,case1,values)]/p(case0)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static Bernoulli Case0AverageConditional<TExit>([SkipIfAllUniform] IList<TExit> values)
        {
            // must takes values as input to distinguish from the other overload.
            return Bernoulli.Uniform();
        }

        /// <summary>EP message to <c>case1</c>.</summary>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>case1</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>case1</c> as the random arguments are varied. The formula is <c>proj[p(case1) sum_(values) p(values) factor(exitTwo,case0,case1,values)]/p(case1)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static Bernoulli Case1AverageConditional<TExit>([SkipIfAllUniform] IList<TExit> values)
        {
            return Bernoulli.Uniform();
        }

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
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitTwoAverageConditional<TExit>(
            TExit exitTwo, Bernoulli case0, Bernoulli case1, [SkipIfAllUniform] IList<TExit> values, TExit result)
            where TExit : SettableTo<TExit>, ICloneable, SettableToProduct<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            result.SetToProduct(exitTwo, values[0]);
            double scale = case0.LogOdds;
            double resultScale = scale;
            // TODO: use pre-allocated buffer
            TExit product = (TExit)exitTwo.Clone();
            product.SetToProduct(exitTwo, values[1]);
            scale = case1.LogOdds;
            double shift = Math.Max(resultScale, scale);
            // avoid (-Infinity) - (-Infinity)
            if (Double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
            }
            else
            {
                result.SetToSum(Math.Exp(resultScale - shift), result, Math.Exp(scale - shift), product);
                resultScale = MMath.LogSumExp(resultScale, scale);
            }
            result.SetToRatio(result, exitTwo);
            return result;
        }

        /// <summary>EP message to <c>case0</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <returns>The outgoing EP message to the <c>case0</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>case0</c> as the random arguments are varied. The formula is <c>proj[p(case0) sum_(exitTwo,values) p(exitTwo,values) factor(exitTwo,case0,case1,values)]/p(case0)</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TExitDomain">The domain of the variable exiting the gate.</typeparam>
        public static Bernoulli Case0AverageConditional<TExit, TExitDomain>(TExit exitTwo, IList<TExitDomain> values)
            where TExit : CanGetLogProb<TExitDomain>
        {
            return Bernoulli.FromLogOdds(exitTwo.GetLogProb(values[0]));
        }

        /// <summary>EP message to <c>case1</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <returns>The outgoing EP message to the <c>case1</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>case1</c> as the random arguments are varied. The formula is <c>proj[p(case1) sum_(exitTwo,values) p(exitTwo,values) factor(exitTwo,case0,case1,values)]/p(case1)</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TExitDomain">The domain of the variable exiting the gate.</typeparam>
        public static Bernoulli Case1AverageConditional<TExit, TExitDomain>(TExit exitTwo, IList<TExitDomain> values)
            where TExit : CanGetLogProb<TExitDomain>
        {
            return Bernoulli.FromLogOdds(exitTwo.GetLogProb(values[1]));
        }

        /// <summary>EP message to <c>exitTwo</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>.</param>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exitTwo</c> as the random arguments are varied. The formula is <c>proj[p(exitTwo) sum_(case0,case1,values) p(case0,case1,values) factor(exitTwo,case0,case1,values)]/p(exitTwo)</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TExitDomain">The domain of the variable exiting the gate.</typeparam>
        public static TExit ExitTwoAverageConditional<TExit, TExitDomain>(
            TExit exitTwo, Bernoulli case0, Bernoulli case1, IList<TExitDomain> values, TExit result)
            where TExit : ICloneable, HasPoint<TExitDomain>, SettableToWeightedSum<TExit>
        {
            result.Point = values[0];
            double scale = case0.LogOdds;
            double resultScale = scale;
            // TODO: overload SetToSum to accept constants.
            TExit product = (TExit)exitTwo.Clone();
            product.Point = values[1];
            scale = case1.LogOdds;
            double shift = Math.Max(resultScale, scale);
            // avoid (-Infinity) - (-Infinity)
            if (Double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
                // do nothing
            }
            else
            {
                result.SetToSum(Math.Exp(resultScale - shift), result, Math.Exp(scale - shift), product);
                resultScale = MMath.LogSumExp(resultScale, scale);
            }
            return result;
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>.</param>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <param name="to_exitTwo">Outgoing message to <c>exitTwo</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static double AverageLogFactor<TExit>(
            TExit exitTwo, Bernoulli case0, Bernoulli case1, IList<TExit> values, [Fresh] TExit to_exitTwo)
            where TExit : ICloneable, SettableToProduct<TExit>,
                SettableToPower<TExit>, CanGetAverageLog<TExit>,
                SettableToUniform, SettableTo<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            // cancel the evidence message from the child variable's child factors
            //T to_exit = (T)exitTwo.Clone();
            //to_exit = ExitTwoAverageLogarithm(case0, case1, values, to_exit);
            return -to_exitTwo.GetAverageLog(exitTwo);
        }

        /// <summary>VMP message to <c>values</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>values</c> with <c>exitTwo</c> integrated out. The formula is <c>sum_exitTwo p(exitTwo) factor(exitTwo,case0,case1,values)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exitTwo" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageLogarithm<TExit, TResultList>([SkipIfUniform, Trigger] TExit exitTwo, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            // result is always exit (messages into a gate are unchanged)
            result.SetAllElementsTo(exitTwo);
            return result;
        }

        /// <summary>VMP message to <c>case0</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>case0</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>case0</c>. Because the factor is deterministic, <c>exitTwo</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(values) p(values) log(sum_exitTwo p(exitTwo) factor(exitTwo,case0,case1,values)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static Bernoulli Case0AverageLogarithm<TExit>(TExit exitTwo, [SkipIfAllUniform, Proper] IList<TExit> values)
            where TExit : CanGetAverageLog<TExit>
        {
            return Bernoulli.FromLogOdds(values[0].GetAverageLog(exitTwo));
        }

        /// <summary>VMP message to <c>case1</c>.</summary>
        /// <param name="exitTwo">Incoming message from <c>exitTwo</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>case1</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>case1</c>. Because the factor is deterministic, <c>exitTwo</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(values) p(values) log(sum_exitTwo p(exitTwo) factor(exitTwo,case0,case1,values)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static Bernoulli Case1AverageLogarithm<TExit>(TExit exitTwo, [SkipIfAllUniform, Proper] IList<TExit> values)
            where TExit : CanGetAverageLog<TExit>
        {
            return Bernoulli.FromLogOdds(values[1].GetAverageLog(exitTwo));
        }

        /// <summary>VMP message to <c>exitTwo</c>.</summary>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exitTwo</c> as the random arguments are varied. The formula is <c>proj[sum_(case0,case1,values) p(case0,case1,values) factor(exitTwo,case0,case1,values)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitTwoAverageLogarithm<TExit>(
            Bernoulli case0, Bernoulli case1, [SkipIfAllUniform, Proper] IList<TExit> values, TExit result)
            where TExit : ICloneable, SettableToProduct<TExit>,
                SettableToPower<TExit>, CanGetAverageLog<TExit>,
                SettableToUniform, SettableTo<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            // result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
            TExit uniform = (TExit)result.Clone();
            uniform.SetToUniform();
            return ExitTwoAverageConditional<TExit>(uniform, case0, case1, values, result);
        }
    }


    /// <summary>Provides outgoing messages for <see cref="Gate.ExitRandom{T}(bool[], T[])" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Gate), "ExitRandom<>")]
    [Quality(QualityBand.Mature)]
    public static class GateExitRandomOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Exit,cases,values))</c>.</para>
        /// </remarks>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>values</c>.</summary>
        /// <param name="exit">Incoming message from <c>Exit</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>values</c> as the random arguments are varied. The formula is <c>proj[p(values) sum_(Exit) p(Exit) factor(Exit,cases,values)]/p(values)</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit[] ValuesAverageConditional<TExit>([IsReturnedInEveryElement] TExit exit, TExit[] result)
        {
            for (int i = 0; i < result.Length; i++)
                result[i] = exit;
            return result;
        }

        /// <summary>EP message to <c>values</c>.</summary>
        /// <param name="exit">Incoming message from <c>Exit</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>values</c> as the random arguments are varied. The formula is <c>proj[p(values) sum_(Exit) p(Exit) factor(Exit,cases,values)]/p(values)</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageConditional<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            // result is always exit (messages into a gate are unchanged)
            return ValuesAverageLogarithm(exit, result);
        }

        /// <summary>EP message to <c>cases</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>cases</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        [Skip]
        public static TResultList CasesAverageConditional<TResultList>(TResultList result)
            where TResultList : SettableToUniform
        {
            return CasesAverageLogarithm<TResultList>(result);
        }

        /// <summary>EP message to <c>Exit</c>.</summary>
        /// <param name="cases">Constant value for <c>cases</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>Exit</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Exit</c> as the random arguments are varied. The formula is <c>proj[p(Exit) sum_(values) p(values) factor(Exit,cases,values)]/p(Exit)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitAverageConditional<TExit>(bool[] cases, [SkipIfAllUniform] IList<TExit> values)
        {
            for (int i = 0; i < cases.Length; i++)
            {
                if (cases[i])
                    return values[i];
            }
            // cases was entirely false
            throw new ArgumentException("cases is all false");
        }

        ///// <summary>
        ///// Gibbs message to 'Exit'
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <typeparam name="TDomain"></typeparam>
        ///// <param name="cases"></param>
        ///// <param name="values"></param>
        ///// <param name="result"></param>
        ///// <returns></returns>
        //public static T ExitAverageConditional<T, TDomain>(IList<bool> cases, IList<TDomain> values, T result)
        //    where T : IDistribution<TDomain>
        //{
        //    for (int i = 0; i < cases.Count; i++)
        //    {
        //        if (cases[i]) return Distribution.SetPoint<T,TDomain>(result, values[i]);
        //    }
        //    // cases was entirely false
        //    throw new ArgumentException("cases is all false");
        //}

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Exit,cases,values))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>values</c>.</summary>
        /// <param name="exit">Incoming message from <c>Exit</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>values</c>. The formula is <c>exp(sum_(Exit) p(Exit) log(factor(Exit,cases,values)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageLogarithm<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            // result is always exit (messages into a gate are unchanged)
            result.SetAllElementsTo(exit);
            return result;
        }

        /// <summary>VMP message to <c>cases</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>cases</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        [Skip]
        public static TResultList CasesAverageLogarithm<TResultList>(TResultList result)
            where TResultList : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>Exit</c>.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="values">Incoming message from <c>values</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Exit</c>. The formula is <c>exp(sum_(cases,values) p(cases,values) log(factor(Exit,cases,values)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="values" /> is not a proper distribution.</exception>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitAverageLogarithm<TExit>(IList<Bernoulli> cases, [SkipIfAllUniform, Proper] IList<TExit> values, TExit result)
            where TExit : ICloneable, SettableToProduct<TExit>,
                SettableToPower<TExit>, CanGetAverageLog<TExit>,
                SettableToUniform, SettableTo<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else
            {
                // result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
                result.SetToPower(values[0], Math.Exp(cases[0].LogOdds));
                if (cases.Count > 1)
                {
                    // TODO: use pre-allocated buffer
                    TExit power = (TExit)result.Clone();
                    for (int i = 1; i < cases.Count; i++)
                    {
                        power.SetToPower(values[i], Math.Exp(cases[i].LogOdds));
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }

        /// <summary />
        /// <param name="values">Incoming message from <c>values</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static TExit ExitAverageLogarithmInit<TExit>([IgnoreDependency] IList<TExit> values)
            where TExit : ICloneable
        {
            return (TExit)values[0].Clone();
        }
    }
}
