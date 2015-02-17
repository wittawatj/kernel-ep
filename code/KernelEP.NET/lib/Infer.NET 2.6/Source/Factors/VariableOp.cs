// (C) Copyright 2009-2010 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Variable{T}(T, out T)" /></description></item><item><description><see cref="Factor.VariableInit{T}(T, T, out T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Variable<>", Default = true)]
    [FactorMethod(typeof(Factor), "VariableInit<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class VariableOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(use,def,marginal))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>marginal</c>.</summary>
        /// <param name="use">Incoming message from <c>use</c>.</param>
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>marginal</c> as the random arguments are varied. The formula is <c>proj[p(marginal) sum_(use,def) p(use,def) factor(use,def,marginal)]/p(marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] T use, T def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetToProduct(def, use);
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T MarginalAverageConditionalInit<T>([IgnoreDependency] T def)
            where T : ICloneable
        {
            return (T)def.Clone();
        }

        /// <summary>EP message to <c>use</c>.</summary>
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <returns>The outgoing EP message to the <c>use</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>use</c> as the random arguments are varied. The formula is <c>proj[p(use) sum_(def) p(def) factor(use,def,marginal)]/p(use)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T UseAverageConditional<T>([IsReturned] T Def)
        {
            return Def;
        }

        /// <summary>EP message to <c>def</c>.</summary>
        /// <param name="use">Incoming message from <c>use</c>.</param>
        /// <returns>The outgoing EP message to the <c>def</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>def</c> as the random arguments are varied. The formula is <c>proj[p(def) sum_(use) p(use) factor(use,def,marginal)]/p(def)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T DefAverageConditional<T>([IsReturned] T use)
        {
            return use;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.VariableGibbs{T}(T, out T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "VariableGibbs<>")]
    [Quality(QualityBand.Preview)]
    public static class VariableGibbsOp
    {
        /// <summary>
        /// Gibbs evidence
        /// </summary>
        /// <returns></returns>
        public static double GibbsEvidence<TDist, T>(TDist Use, TDist Def, GibbsMarginal<TDist, T> marginal)
            where TDist : IDistribution<T>, Sampleable<T>, CanGetLogAverageOf<TDist>
        {
            return Def.GetLogAverageOf(Use) - Def.GetLogProb(marginal.LastSample) - Use.GetLogProb(marginal.LastSample);
        }

        /// <summary>
        /// Gibbs message to 'Marginal'.
        /// </summary>
        /// <param name="Use">Incoming message from 'use'.</param>
        /// <param name="Def">Incoming message from 'def'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="to_marginal"></param>
        /// <remarks><para>
        /// The outgoing message is the product of 'Def' and 'Uses' messages.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Def"/> is not a proper distribution</exception>
        [Stochastic]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist, T>(
            TDist Use,
            [SkipIfUniform] TDist Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.SetToProduct(Def, Use);
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        [Stochastic]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist, T>(
            T Use,
            [SkipIfUniform] TDist Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Use;
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary>
        /// Gibbs sample message to 'Use'
        /// </summary>
        /// <param name="marginal">Incoming message from 'marginal'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the current Gibbs sample.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="marginal"/> is not a proper distribution</exception>
        public static T UseGibbs<TDist, T>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return marginal.LastSample;
        }

        /// <summary>
        /// Gibbs distribution message to 'Def'
        /// </summary>
        /// <param name="Def">Incoming message from 'def'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the product of the 'Def' message with all 'Uses' messages except the current
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Def"/> is not a proper distribution</exception>
        public static TDist UseGibbs<TDist, T>([IsReturned] TDist Def, TDist result)
            where TDist : SettableTo<TDist>
        {
            result.SetTo(Def);
            return result;
        }

        /// <summary>
        /// Gibbs sample message to 'Def'
        /// </summary>
        /// <param name="marginal">Incoming message from 'marginal'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the current Gibbs sample.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="marginal"/> is not a proper distribution</exception>
        public static T DefGibbs<TDist, T>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return marginal.LastSample;
        }

        public static TDist DefGibbs<TDist, T>([IsReturned] TDist Use, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            result.SetTo(Use);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.VariableMax{T}(T, out T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "VariableMax<>")]
    [Quality(QualityBand.Preview)]
    public static class VariableMaxOp
    {
        /// <summary />
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T UseMaxConditional<T>(T Def, T result)
            where T : SettableTo<T>
        {
            result.SetTo(Def);
            if (result is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)result).SetMaxToZero();
            return result;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T UseMaxConditionalInit<T>([IgnoreDependency] T Def)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <summary />
        /// <param name="Use">Incoming message from <c>use</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T DefMaxConditional<T>([IsReturned] T Use)
        {
            return Use;
        }

        /// <summary />
        /// <param name="Use">Incoming message from <c>use</c>.</param>
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [MultiplyAll]
        public static T MarginalMaxConditional<T>([NoInit] T Use, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = VariableOp.MarginalAverageConditional<T>(Use, Def, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T MarginalMaxConditionalInit<T>([IgnoreDependency] T def)
            where T : ICloneable
        {
            return (T)def.Clone();
        }
    }

#if true
    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Variable{T}(T, out T)" /></description></item><item><description><see cref="Factor.VariableInit{T}(T, T, out T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Variable<>", Default = true)]
    [FactorMethod(typeof(Factor), "VariableInit<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class VariableVmpOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(use,def,marginal))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static double AverageLogFactor<T>([SkipIfUniform] T to_marginal /*, [IgnoreDependency] T def*/)
            where T : CanGetAverageLog<T>
        {
            return -to_marginal.GetAverageLog(to_marginal);
        }

        /// <summary>VMP message to <c>marginal</c>.</summary>
        /// <param name="use">Incoming message from <c>use</c>.</param>
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>marginal</c>. The formula is <c>exp(sum_(use,def) p(use,def) log(factor(use,def,marginal)))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageLogarithm<T>([NoInit] T use, T def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetToProduct(def, use);
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T MarginalAverageLogarithmInit<T>([IgnoreDependency] T def)
            where T : ICloneable
        {
            return (T)def.Clone();
        }

        /// <summary>VMP message to <c>use</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>use</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T UseAverageLogarithm<T>([IsReturned] T to_marginal, T result)
            where T : SettableTo<T>
        {
            result.SetTo(to_marginal);
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T UseAverageLogarithmInit<T>([IgnoreDependency] T def)
            where T : ICloneable
        {
            return (T)def.Clone();
        }

        /// <summary>VMP message to <c>def</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>def</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T DefAverageLogarithm<T>([IsReturned] T to_marginal, T result)
            where T : SettableTo<T>
        {
            result.SetTo(to_marginal);
            return result;
        }
    }
#else
    /// <summary>
    /// Provides outgoing VMP messages for <see cref="Factor.Variable&lt;T&gt;"/>, given random arguments to the function.
    /// </summary>
	[FactorMethod(typeof(Factor), "Variable<>")]
	[FactorMethod(typeof(Factor), "VariableInit<>", Default=false)]
	[Buffers("marginalB")]
	[Quality(QualityBand.Preview)]
	public static class VariableVmpBufferOp
	{
		/// <summary>
		/// Evidence message for VMP
		/// </summary>
		/// <param name="marginal">Outgoing message to 'marginal'.</param>
		/// <returns>Average of the factor's log-value across the given argument distributions</returns>
		/// <remarks><para>
		/// The formula for the result is <c>log(factor(Uses,Def,Marginal))</c>.
		/// Adding up these values across all factors and variables gives the log-evidence estimate for VMP.
		/// </para></remarks>
		public static double AverageLogFactor<T>([Fresh, SkipIfUniform] T marginalB, [IgnoreDependency] T def)
			where T : CanGetAverageLog<T>
		{
			return -marginalB.GetAverageLog(marginalB);
		}

		[SkipIfAllUniform]
		[MultiplyAll]
		public static T MarginalB<T>(T use, T def, T result)
				where T : SettableToProduct<T>, SettableTo<T>
		{
			result.SetToProduct(def, use);
			return result;
		}
		//[Skip]
		//public static T MarginalBInit<T>([IgnoreDependency] T def)
		//  where T : ICloneable
		//{
		//  return (T)def.Clone();
		//}

		public static T MarginalAverageLogarithm<T>([IsReturned] T marginalB, T result)
			where T : SettableTo<T>
		{
			result.SetTo(marginalB);
			return result;
		}

		/// <summary>
		/// VMP message to 'use'
		/// </summary>
		/// <param name="marginal">Current 'marginal'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is the factor viewed as a function of 'use' conditioned on the given values.
		/// </para></remarks>
		public static T UseAverageLogarithm<T>([IsReturned] T marginalB, T result)
			where T : SettableTo<T>
		{
			result.SetTo(marginalB);
			return result;
		}

		/// <summary>
		/// VMP message to 'def'
		/// </summary>
		/// <param name="marginal">Current 'marginal'.</param>
		/// <param name="result">Modified to contain the outgoing message</param>
		/// <returns><paramref name="result"/></returns>
		/// <remarks><para>
		/// The outgoing message is the factor viewed as a function of 'def' conditioned on the given values.
		/// </para></remarks>
		public static T DefAverageLogarithm<T>([IsReturned] T marginalB, T result)
			where T : SettableTo<T>
		{
			result.SetTo(marginalB);
			return result;
		}
	}
	/// <summary>
	/// Provides outgoing VMP messages for <see cref="Factor.Variable&lt;T&gt;"/>, given random arguments to the function.
	/// </summary>
	[FactorMethod(typeof(Factor), "Variable<>")]
	[Buffers("marginalB")]
	[Quality(QualityBand.Preview)]
	public static class VariableNoInitVmpBufferOp
	{
		[Skip]
		public static T MarginalBInit<T>([IgnoreDependency] T def)
			where T : ICloneable
		{
			return (T)def.Clone();
		}
	}

	/// <summary>
	/// Provides outgoing VMP messages for <see cref="Factor.Variable&lt;T&gt;"/>, given random arguments to the function.
	/// </summary>
	[FactorMethod(typeof(Factor), "VariableInit<>", Default=true)]
	[Buffers("marginalB")]
	[Quality(QualityBand.Preview)]
	public static class VariableInitVmpBufferOp
	{
		// def is included to get its type as a constraint.  not needed if we could bind on return type.
		public static T MarginalBInit<T>([IgnoreDependency] T def, [SkipIfUniform] T init)
			where T : ICloneable
		{
			return (T)init.Clone();
		}
	}
#endif

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.DerivedVariable{T}(T, out T)" /></description></item><item><description><see cref="Factor.DerivedVariableInit{T}(T, T, out T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "DerivedVariable<>", Default = true)]
    [FactorMethod(typeof(Factor), "DerivedVariableInit<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class DerivedVariableOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(use,def,marginal))</c>.</para>
        /// </remarks>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(use,def,marginal))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>marginal</c>.</summary>
        /// <param name="Use">Incoming message from <c>use</c>.</param>
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>marginal</c> as the random arguments are varied. The formula is <c>proj[p(marginal) sum_(use,def) p(use,def) factor(use,def,marginal)]/p(marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] T Use, T Def, T result)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(Def, Use);
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T MarginalAverageConditionalInit<T>([IgnoreDependency] T def)
            where T : ICloneable
        {
            return (T)def.Clone();
        }

        /// <summary>EP message to <c>use</c>.</summary>
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <returns>The outgoing EP message to the <c>use</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>use</c> as the random arguments are varied. The formula is <c>proj[p(use) sum_(def) p(def) factor(use,def,marginal)]/p(use)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T UseAverageConditional<T>([IsReturned] T Def)
        {
            return Def;
        }

        /// <summary>EP message to <c>def</c>.</summary>
        /// <param name="Use">Incoming message from <c>use</c>.</param>
        /// <returns>The outgoing EP message to the <c>def</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>def</c> as the random arguments are varied. The formula is <c>proj[p(def) sum_(use) p(use) factor(use,def,marginal)]/p(def)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T DefAverageConditional<T>([IsReturned] T Use)
        {
            return Use;
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.DerivedVariableInitGibbs{T}(T, T, out T)" /></description></item><item><description><see cref="Factor.DerivedVariableGibbs{T}(T, out T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "DerivedVariableGibbs<>")]
    [FactorMethod(typeof(Factor), "DerivedVariableInitGibbs<>")]
    [Quality(QualityBand.Preview)]
    public static class DerivedVariableGibbsOp
    {
        #region Gibbs messages

        /// <summary>
        /// Evidence message for Gibbs.
        /// </summary>
        [Skip]
        public static double GibbsEvidence()
        {
            return 0.0;
        }

        /// <summary>
        /// Gibbs sample message to 'Uses'
        /// </summary>
        /// <typeparam name="TDist">Gibbs marginal type</typeparam>
        /// <typeparam name="T">Domain type</typeparam>
        /// <param name="marginal">The Gibbs marginal</param>
        /// <param name="def"></param>
        /// <param name="result">Result</param>
        /// <returns></returns>
        /// <remarks><para>
        /// The outgoing message is the current Gibbs sample.
        /// </para></remarks>
        public static T UseGibbs<TDist, T>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, TDist def, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            // This method must depend on Def, even though Def isn't used, in order to get the right triggers
            return marginal.LastSample;
        }

        /// <summary>
        /// Gibbs sample message to 'Uses'
        /// </summary>
        /// <typeparam name="T">Domain type</typeparam>
        /// <param name="def"></param>
        /// <param name="result">Result</param>
        /// <returns></returns>
        /// <remarks><para>
        /// The outgoing message is the current Gibbs sample.
        /// </para></remarks>
        public static T UseGibbs<T>([IsReturned] T def, T result)
        {
            return def;
        }

        /// <summary>
        /// Gibbs distribution message to 'Def'
        /// </summary>
        /// <typeparam name="TDist">Distribution type</typeparam>
        /// <typeparam name="T">Domain type</typeparam>
        /// <param name="Use">Incoming message from 'Use'.</param>
        /// <remarks><para>
        /// The outgoing message is the product of all the 'Use' messages.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Use"/> is not a proper distribution</exception>
        public static TDist DefGibbs<TDist, T>([IsReturned] TDist Use)
        {
            return Use;
        }

        public static T DefGibbs<TDist, T>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return marginal.LastSample;
        }

        /// <summary>
        /// Gibbs message to 'Marginal' for distribution Def
        /// </summary>
        /// <param name="Use">Incoming message from 'Use'.</param>
        /// <param name="Def">Incoming message from 'Def'.</param>
        /// <param name="to_marginal">Previous outgoing message to 'marginal'.</param>
        /// <returns><paramref name="to_marginal"/></returns>
        /// <remarks><para>
        /// The outgoing message is the product of 'Def' and 'Use' messages.
        /// </para></remarks>
        [Stochastic]
        [SkipIfAllUniform]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist, T>(
            TDist Use,
            [SkipIfUniform] TDist Def,
            GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.SetToProduct(Def, Use);
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary>
        /// Gibbs message to 'Marginal' for sample Def
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="Def"></param>
        /// <param name="to_marginal">Previous outgoing message to 'marginal'.</param>
        /// <returns><paramref name="to_marginal"/></returns>
        [Stochastic] // must be labelled Stochastic to get correct schedule, even though it isn't Stochastic
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist, T>(
            T Def,
            GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Def;
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary>
        /// Gibbs message to 'Marginal' for sample Use
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="Use"></param>
        /// <param name="Def"></param>
        /// <param name="to_marginal">Previous outgoing message to 'marginal'.</param>
        /// <returns><paramref name="to_marginal"/></returns>
        [Stochastic] // must be labelled Stochastic to get correct schedule, even though it isn't Stochastic
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist, T>(
            T Use, [IgnoreDependency] TDist Def,
            GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Use;
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        #endregion
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.DerivedVariableInitVmp{T}(T, T, out T)" /></description></item><item><description><see cref="Factor.DerivedVariableVmp{T}(T, out T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "DerivedVariableVmp<>", Default = true)]
    [FactorMethod(typeof(Factor), "DerivedVariableInitVmp<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class DerivedVariableVmpOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(use,def,init,marginal))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>marginal</c>.</summary>
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>marginal</c>. The formula is <c>exp(sum_(def) p(def) log(factor(use,def,init,marginal)))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        public static T MarginalAverageLogarithm<T, TDef>([IsReturned] TDef Def, T result)
            where T : SettableTo<TDef>
        {
            result.SetTo(Def);
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T MarginalAverageLogarithmInit<T>([IgnoreDependency] T def)
            where T : ICloneable
        {
            return (T)def.Clone();
        }

        /// <summary>VMP message to <c>use</c>.</summary>
        /// <param name="Def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>use</c> as the random arguments are varied. The formula is <c>proj[sum_(def) p(def) factor(use,def,init,marginal)]</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        /// <typeparam name="TDef"></typeparam>
        public static T UseAverageLogarithm<T, TDef>([IsReturned] TDef Def, T result)
            where T : SettableTo<TDef>
        {
            result.SetTo(Def);
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static T UseAverageLogarithmInit<T>([IgnoreDependency] T def)
            where T : ICloneable
        {
            return (T)def.Clone();
        }

        /// <summary>VMP message to <c>def</c>.</summary>
        /// <param name="Use">Incoming message from <c>use</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>def</c> with <c>use</c> integrated out. The formula is <c>sum_use p(use) factor(use,def,init,marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the marginal of the variable.</typeparam>
        public static T DefAverageLogarithm<T>(
            [IsReturned] T Use, T result) // must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
            where T : SettableTo<T>
        {
            result.SetTo(Use);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Variable{T}(T, out T)" /></description></item><item><description><see cref="Factor.VariableInit{T}(T, T, out T)" /></description></item><item><description><see cref="Factor.DerivedVariable{T}(T, out T)" /></description></item><item><description><see cref="Factor.DerivedVariableInit{T}(T, T, out T)" /></description></item><item><description><see cref="Factor.DerivedVariableInitVmp{T}(T, T, out T)" /></description></item><item><description><see cref="Factor.DerivedVariableVmp{T}(T, out T)" /></description></item><item><description><see cref="Factor.VariablePoint{T}(T, out T)" /></description></item></list>, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable.</typeparam>
    [FactorMethod(typeof(Factor), "Variable<>", Default = false)]
    [FactorMethod(typeof(Factor), "VariableInit<>", Default = false)]
    [FactorMethod(typeof(Factor), "DerivedVariable<>", Default = false)]
    [FactorMethod(typeof(Factor), "DerivedVariableInit<>", Default = false)]
    [FactorMethod(typeof(Factor), "DerivedVariableVmp<>", Default = false)]
    [FactorMethod(typeof(Factor), "DerivedVariableInitVmp<>", Default = false)]
    [FactorMethod(typeof(Factor), "VariablePoint<>", Default = true)]
    [Quality(QualityBand.Experimental)]
    public static class VariablePointOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="use">Incoming message from <c>use</c>.</param>
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(use,def) p(use,def) factor(use,def,marginal) / sum_use p(use) messageTo(use))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static double LogEvidenceRatio<TDist>(TDist use, TDist def, TDist to_marginal)
            where TDist : CanGetLogAverageOf<TDist>
        {
            //return def.GetLogAverageOf(to_marginal);
            return def.GetLogAverageOf(use) - use.GetLogAverageOf(to_marginal);
        }

        /// <summary>EP message to <c>marginal</c>.</summary>
        /// <param name="use">Incoming message from <c>use</c>.</param>
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>marginal</c> as the random arguments are varied. The formula is <c>proj[p(marginal) sum_(use,def) p(use,def) factor(use,def,marginal)]/p(marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        public static TDist MarginalAverageConditional<TDist>([NoInit] TDist use, TDist def, TDist result)
            where TDist : SettableToProduct<TDist>, HasPoint<T>, CanGetMode<T>
        {
            result.SetToProduct(def, use);
            result.Point = result.GetMode();
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static TDist MarginalAverageConditionalInit<TDist>([IgnoreDependency] TDist def)
            where TDist : ICloneable
        {
            return (TDist)def.Clone();
        }

        /// <summary>EP message to <c>use</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns>The outgoing EP message to the <c>use</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>use</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist UseAverageConditional<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }

        /// <summary>EP message to <c>def</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns>The outgoing EP message to the <c>def</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>def</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist DefAverageConditional<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(use,def,marginal))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>marginal</c>.</summary>
        /// <param name="use">Incoming message from <c>use</c>.</param>
        /// <param name="def">Incoming message from <c>def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>marginal</c>. The formula is <c>exp(sum_(use,def) p(use,def) log(factor(use,def,marginal)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        public static TDist MarginalAverageLogarithm<TDist>([NoInit] TDist use, TDist def, TDist result)
            where TDist : SettableToProduct<TDist>, HasPoint<T>, CanGetMode<T>
        {
            return MarginalAverageConditional(use, def, result);
        }

        /// <summary>VMP message to <c>use</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns>The outgoing VMP message to the <c>use</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>use</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist UseAverageLogarithm<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }

        /// <summary>VMP message to <c>def</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns>The outgoing VMP message to the <c>def</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>def</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist DefAverageLogarithm<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }
    }
}
