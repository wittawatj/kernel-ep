// (C) Copyright 2008 Microsoft Research Cambridge

using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;

namespace MicrosoftResearch.Infer.Factors
{
    /// <summary>
    /// Factors for handling gates.
    /// </summary>
    [Hidden]
    public static class Gate
    {

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Enter factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="selector"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static T[] Enter<T>(int selector, [IsReturnedInEveryElement] T value)
        {
            throw new NotImplementedException();
            T[] result = new T[2];
            for (int i = 0; i < 2; i++)
            {
                result[i] = value;
            }
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        public static T[] Enter<T>(bool selector, [IsReturnedInEveryElement] T value)
        {
            T[] result = new T[2];
            for (int i = 0; i < 2; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Enter partial factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="selector"></param>
        /// <param name="value"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static T[] EnterPartial<T>(int selector, [IsReturnedInEveryElement] T value, int[] indices)
        {
            T[] result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        public static T[] EnterPartial<T>(bool selector, [IsReturnedInEveryElement] T value, int[] indices)
        {
            T[] result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Enter partial factor with two cases
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="case0"></param>
        /// <param name="case1"></param>
        /// <param name="value"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static T[] EnterPartialTwo<T>(bool case0, bool case1, [IsReturnedInEveryElement] T value, int[] indices)
        {
            T[] result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Enter one factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="selector"></param>
        /// <param name="value"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public static T EnterOne<T>(int selector, [IsReturned] T value, int index)
        {
            return value;
        }

        /// <summary>
        /// Exit factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="cases"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public static T Exit<T>(bool[] cases, T[] values)
        {
            for (int i = 0; i < cases.Length; i++)
                if (cases[i])
                    return values[i];

            throw new ApplicationException("Exit factor: no case is true");
        }

        /// <summary>
        /// Exit factor with two cases
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="case0"></param>
        /// <param name="case1"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public static T ExitTwo<T>(bool case0, bool case1, T[] values)
        {
            if (case0)
                return values[0];
            else if (case1)
                return values[1];

            throw new ApplicationException("ExitTwo factor: neither case is true");
        }

        /// <summary>
        /// Exit random factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="cases"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("Exit", "cases", "values")]
        public static T ExitRandom<T>(bool[] cases, T[] values)
        {
            return Exit(cases, values);
        }

        /// <summary>
        /// Exit random factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="case0"></param>
        /// <param name="case1"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("Exit", "cases", "values")]
        public static T ExitRandomTwo<T>(bool case0, bool case1, T[] values)
        {
            if (case0)
                return values[0];
            else if (case1)
                return values[1];

            throw new ApplicationException("ExitTwo factor: neither case is true");
        }

#if true
        /// <summary>
        /// Exiting variable
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="Def"></param>
        /// <param name="Marginal"></param>
        /// <returns></returns>
        [ParameterNames("Use", "Def", "Marginal")]
        public static T ExitingVariable<T>(T Def, out T Marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Replicate an exiting variable
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="Def"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        [ParameterNames("Uses", "Def", "count")]
        public static T[] ReplicateExiting<T>(T Def, int count)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }
#else
    /// <summary>
    /// Exiting variable factor
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="Def"></param>
    /// <param name="Marginal"></param>
    /// <returns></returns>
		[ParameterNames("Uses", "Def", "Marginal")]
		public static T[] ExitingVariable<T>(T Def, T Marginal)
		{
			throw new InvalidOperationException("Should never be called with deterministic arguments");
		}
#endif

        /// <summary>
        /// Boolean cases factor
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool[] Cases(bool b)
        {
            bool[] result = new bool[2];
            result[0] = b;
            result[1] = !b;
            return result;
        }

        /// <summary>
        /// Boolean cases factor expanded into elements
        /// </summary>
        /// <param name="b"></param>
        /// <param name="case0">case 0 (true)</param>
        /// <param name="case1">case 1 (false)</param>
        /// <returns></returns>
        public static void CasesBool(bool b, out bool case0, out bool case1)
        {
            case0 = b;
            case1 = !b;
        }

        // TODO: fix bug which prevents this being called 'Cases'
        /// <summary>
        /// Integer cases factor
        /// </summary>
        /// <param name="i">index</param>
        /// <param name="count">number of cases</param>
        /// <returns></returns>
        public static bool[] CasesInt(int i, int count)
        {
            bool[] result = new bool[count];
            for (int j = 0; j < count; j++)
                result[j] = false;
            result[i] = true;
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.Cases(bool)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Gate), "Cases", typeof(bool))]
    [Quality(QualityBand.Mature)]
    public static class CasesOp
    {

        /// <summary>EP message to <c>cases</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>cases</c> as the random arguments are varied. The formula is <c>proj[p(cases) sum_(b) p(b) factor(cases,b)]/p(cases)</c>.</para>
        /// </remarks>
        /// <typeparam name="BernoulliList">The type of the outgoing message.</typeparam>
        public static BernoulliList CasesAverageConditional<BernoulliList>(Bernoulli b, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            // result.LogOdds = [log p(b=true), log p(b=false)]
            if (result.Count != 2)
                throw new ArgumentException("result.Count != 2");
            result[0] = Bernoulli.FromLogOdds(b.GetLogProbTrue());
            result[1] = Bernoulli.FromLogOdds(b.GetLogProbFalse());
            return result;
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static DistributionStructArray<Bernoulli, bool> CasesAverageConditionalInit()
        {
            return new DistributionStructArray<Bernoulli, bool>(2);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(cases) p(cases) factor(cases,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="cases" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageConditional([SkipIfUniform] IList<Bernoulli> cases)
        {
            // result = p(b=true) / (p(b=true) + p(b=false))
            //        = 1 / (1 + p(b=false)/p(b=true))
            //        = 1 / (1 + exp(-(log p(b=true) - log p(b=false)))
            // where cases[0].LogOdds = log p(b=true)
            //       cases[1].LogOdds = log p(b=false)
            if (cases[0].LogOdds == cases[1].LogOdds) // avoid (-Infinity) - (-Infinity)
            {
                if (Double.IsNegativeInfinity(cases[0].LogOdds) && Double.IsNegativeInfinity(cases[1].LogOdds))
                    throw new AllZeroException();
                return new Bernoulli();
            }
            else
            {
                return Bernoulli.FromLogOdds(cases[0].LogOdds - cases[1].LogOdds);
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(cases,b) p(cases,b) factor(cases,b) / sum_cases p(cases) messageTo(cases))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(IList<Bernoulli> cases, Bernoulli b)
        {
            // result = log (p(data|b=true) p(b=true) + p(data|b=false) p(b=false))
            //          log (p(data|b=true) p(b=true) + p(data|b=false) (1-p(b=true))
            //          log ((p(data|b=true) - p(data|b=false)) p(b=true) + p(data|b=false))
            //          log ((p(data|b=true)/p(data|b=false) - 1) p(b=true) + 1) + log p(data|b=false)
            // where cases[0].LogOdds = log p(data|b=true)
            //       cases[1].LogOdds = log p(data|b=false)
            if (b.IsPointMass)
                return b.Point ? cases[0].LogOdds : cases[1].LogOdds;
            //else return MMath.LogSumExp(cases[0].LogOdds + b.GetLogProbTrue(), cases[1].LogOdds + b.GetLogProbFalse());
            else
            {
                // the common case is when cases[0].LogOdds == cases[1].LogOdds.  we must not introduce rounding error in that case.
                if (cases[0].LogOdds >= cases[1].LogOdds)
                {
                    if (Double.IsNegativeInfinity(cases[1].LogOdds))
                        return cases[0].LogOdds + b.GetLogProbTrue();
                    else
                        return cases[1].LogOdds + MMath.Log1Plus(b.GetProbTrue() * MMath.ExpMinus1(cases[0].LogOdds - cases[1].LogOdds));
                }
                else
                {
                    if (Double.IsNegativeInfinity(cases[0].LogOdds))
                        return cases[1].LogOdds + b.GetLogProbFalse();
                    else
                        return cases[0].LogOdds + MMath.Log1Plus(b.GetProbFalse() * MMath.ExpMinus1(cases[1].LogOdds - cases[0].LogOdds));
                }
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(cases) p(cases) factor(cases,b) / sum_cases p(cases) messageTo(cases))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(IList<Bernoulli> cases, bool b)
        {
            return 0.0;
            //return b ? cases[0].LogOdds : cases[1].LogOdds;
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <summary>VMP message to <c>cases</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>cases</c> as the random arguments are varied. The formula is <c>proj[sum_(b) p(b) factor(cases,b)]</c>.</para>
        /// </remarks>
        /// <typeparam name="BernoulliList">The type of the outgoing message.</typeparam>
        public static BernoulliList CasesAverageLogarithm<BernoulliList>(Bernoulli b, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            return CasesAverageConditional(b, result);
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static DistributionStructArray<Bernoulli, bool> CasesAverageLogarithmInit()
        {
            return new DistributionStructArray<Bernoulli, bool>(2);
        }

        [Skip]
        public static DistributionType CasesDeriv<DistributionType>(DistributionType result)
            where DistributionType : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> with <c>cases</c> integrated out. The formula is <c>sum_cases p(cases) factor(cases,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="cases" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] IList<Bernoulli> cases) // TM: SkipIfAny (rather than SkipIfAll) is important for getting good schedules
        {
            return BAverageConditional(cases);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="cases">Incoming message from <c>cases</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="cases" /> is not a proper distribution.</exception>
        public static double AverageLogFactor([SkipIfUniform] IList<Bernoulli> cases, Bernoulli b)
        {
            double probTrue = b.GetProbTrue();
            return probTrue * cases[0].LogOdds + (1 - probTrue) * cases[1].LogOdds;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.CasesBool(bool, out bool, out bool)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Gate), "CasesBool", typeof(bool), typeof(bool), typeof(bool))]
    [Quality(QualityBand.Experimental)]
    public static class CasesBoolOp
    {
        /// <summary>EP message to <c>case0</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>case0</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>case0</c> as the random arguments are varied. The formula is <c>proj[p(case0) sum_(b) p(b) factor(b,case0,case1)]/p(case0)</c>.</para>
        /// </remarks>
        public static Bernoulli Case0AverageConditional(Bernoulli b)
        {
            return Bernoulli.FromLogOdds(b.GetLogProbTrue());
        }

        /// <summary>EP message to <c>case1</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>case1</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>case1</c> as the random arguments are varied. The formula is <c>proj[p(case1) sum_(b) p(b) factor(b,case0,case1)]/p(case1)</c>.</para>
        /// </remarks>
        public static Bernoulli Case1AverageConditional(Bernoulli b)
        {
            return Bernoulli.FromLogOdds(b.GetLogProbFalse());
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(case0,case1) p(case0,case1) factor(b,case0,case1)]/p(b)</c>.</para>
        /// </remarks>
        [SkipIfAllUniform]
        public static Bernoulli BAverageConditional(Bernoulli case0, Bernoulli case1)
        {
            // result = p(b=true) / (p(b=true) + p(b=false))
            //        = 1 / (1 + p(b=false)/p(b=true))
            //        = 1 / (1 + exp(-(log p(b=true) - log p(b=false)))
            // where cases[0].LogOdds = log p(b=true)
            //       cases[1].LogOdds = log p(b=false)
            // avoid (-Infinity) - (-Infinity)
            if (case0.LogOdds == case1.LogOdds)
            {
                if (Double.IsNegativeInfinity(case0.LogOdds) && Double.IsNegativeInfinity(case1.LogOdds))
                    throw new AllZeroException();
                return new Bernoulli();
            }
            else
            {
                return Bernoulli.FromLogOdds(case0.LogOdds - case1.LogOdds);
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(case0,case1,b) p(case0,case1,b) factor(b,case0,case1))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Bernoulli case0, Bernoulli case1, Bernoulli b)
        {
            // result = log (p(data|b=true) p(b=true) + p(data|b=false) p(b=false))
            // where cases[0].LogOdds = log p(data|b=true)
            //       cases[1].LogOdds = log p(data|b=false)
            if (b.IsPointMass)
                return b.Point ? case0.LogOdds : case1.LogOdds;
            else
                return MMath.LogSumExp(case0.LogOdds + b.GetLogProbTrue(), case1.LogOdds + b.GetLogProbFalse());
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <summary>VMP message to <c>case0</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>case0</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>case0</c>. The formula is <c>exp(sum_(b) p(b) log(factor(b,case0,case1)))</c>.</para>
        /// </remarks>
        public static Bernoulli Case0AverageLogarithm(Bernoulli b)
        {
            return Case0AverageConditional(b);
        }

        /// <summary>VMP message to <c>case1</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>case1</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>case1</c>. The formula is <c>exp(sum_(b) p(b) log(factor(b,case0,case1)))</c>.</para>
        /// </remarks>
        public static Bernoulli Case1AverageLogarithm(Bernoulli b)
        {
            return Case1AverageConditional(b);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(case0,case1) p(case0,case1) log(factor(b,case0,case1)))</c>.</para>
        /// </remarks>
        [SkipIfAllUniform]
        public static Bernoulli BAverageLogarithm(Bernoulli case0, Bernoulli case1)
        {
            return BAverageConditional(case0, case1);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="case0">Incoming message from <c>case0</c>.</param>
        /// <param name="case1">Incoming message from <c>case1</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [SkipIfAllUniform("case0", "case1")]
        public static double AverageLogFactor(Bernoulli case0, Bernoulli case1, Bernoulli b)
        {
            double probTrue = b.GetProbTrue();
            return probTrue * case0.LogOdds + (1 - probTrue) * case1.LogOdds;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Gate.CasesInt(int, int)" />, given random arguments to the function.</summary>
    [FactorMethod(new string[] { "Cases", "i", "count" }, typeof(Gate), "CasesInt", typeof(int), typeof(int))]
    [Quality(QualityBand.Mature)]
    public static class IntCasesOp
    {
        /// <summary>EP message to <c>casesInt</c>.</summary>
        /// <param name="i">Incoming message from <c>i</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>casesInt</c> as the random arguments are varied. The formula is <c>proj[p(casesInt) sum_(i) p(i) factor(casesInt,i,count)]/p(casesInt)</c>.</para>
        /// </remarks>
        /// <typeparam name="BernoulliList">The type of the outgoing message.</typeparam>
        public static BernoulliList CasesAverageConditional<BernoulliList>(Discrete i, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            if (result.Count != i.Dimension)
                throw new ArgumentException("result.Count (" + result.Count + ") != i.Dimension (" + i.Dimension + ")");
            for (int j = 0; j < result.Count; j++)
            {
                result[j] = Bernoulli.FromLogOdds(i.GetLogProb(j));
            }
            return result;
        }

        /// <summary />
        /// <param name="i">Incoming message from <c>i</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static DistributionStructArray<Bernoulli, bool> CasesAverageConditionalInit([IgnoreDependency] Discrete i)
        {
            return new DistributionStructArray<Bernoulli, bool>(i.Dimension);
        }

        /// <summary>EP message to <c>i</c>.</summary>
        /// <param name="cases">Incoming message from <c>casesInt</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>i</c> as the random arguments are varied. The formula is <c>proj[p(i) sum_(casesInt) p(casesInt) factor(casesInt,i,count)]/p(i)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="cases" /> is not a proper distribution.</exception>
        public static Discrete IAverageConditional([SkipIfUniform] IList<Bernoulli> cases, Discrete result)
        {
            Vector probs = result.GetWorkspace();
            double max = cases[0].LogOdds;
            for (int j = 1; j < cases.Count; j++)
            {
                if (cases[j].LogOdds > max)
                    max = cases[j].LogOdds;
            }
            // if result.Dimension > cases.Count, the missing cases have LogOdds=0
            if (result.Dimension > cases.Count)
                max = Math.Max(max, 0);
            // avoid (-Infinity) - (-Infinity)
            if (Double.IsNegativeInfinity(max))
                throw new AllZeroException();

            if (probs.Sparsity.IsApproximate)
            {
                var sparseProbs = probs as ApproximateSparseVector;
                probs.SetAllElementsTo(sparseProbs.Tolerance);
            }

            for (int j = 0; j < result.Dimension; j++)
            {
                if (j < cases.Count)
                    probs[j] = Math.Exp(cases[j].LogOdds - max);
                else
                    probs[j] = Math.Exp(0 - max);
            }

            result.SetProbs(probs);
            return result;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="cases">Incoming message from <c>casesInt</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="i">Incoming message from <c>i</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(casesInt,i) p(casesInt,i) factor(casesInt,i,count) / sum_casesInt p(casesInt) messageTo(casesInt))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="cases" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] IList<Bernoulli> cases, Discrete i)
        {
            if (i.IsPointMass)
                return cases[i.Point].LogOdds;
            else
            {
                double[] logOdds = new double[cases.Count];
                for (int j = 0; j < cases.Count; j++)
                {
                    logOdds[j] = cases[j].LogOdds + i.GetLogProb(j);
                }
                return MMath.LogSumExp(logOdds);
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="cases">Incoming message from <c>casesInt</c>.</param>
        /// <param name="i">Constant value for <c>i</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(casesInt) p(casesInt) factor(casesInt,i,count) / sum_casesInt p(casesInt) messageTo(casesInt))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(IList<Bernoulli> cases, int i)
        {
            return cases[i].LogOdds;
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <summary>VMP message to <c>casesInt</c>.</summary>
        /// <param name="i">Incoming message from <c>i</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>casesInt</c> as the random arguments are varied. The formula is <c>proj[sum_(i) p(i) factor(casesInt,i,count)]</c>.</para>
        /// </remarks>
        /// <typeparam name="BernoulliList">The type of the outgoing message.</typeparam>
        public static BernoulliList CasesAverageLogarithm<BernoulliList>(Discrete i, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            return CasesAverageConditional(i, result);
        }

        /// <summary />
        /// <param name="i">Incoming message from <c>i</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static DistributionStructArray<Bernoulli, bool> CasesAverageLogarithmInit([IgnoreDependency] Discrete i)
        {
            return new DistributionStructArray<Bernoulli, bool>(i.Dimension);
        }

        /// <summary>VMP message to <c>i</c>.</summary>
        /// <param name="cases">Incoming message from <c>casesInt</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>i</c> with <c>casesInt</c> integrated out. The formula is <c>sum_casesInt p(casesInt) factor(casesInt,i,count)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="cases" /> is not a proper distribution.</exception>
        public static Discrete IAverageLogarithm([SkipIfUniform] IList<Bernoulli> cases, Discrete result)
        {
            return IAverageConditional(cases, result);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="cases">Incoming message from <c>casesInt</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="i">Incoming message from <c>i</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="cases" /> is not a proper distribution.</exception>
        public static double AverageLogFactor([SkipIfAllUniform] IList<Bernoulli> cases, Discrete i)
        {
            double sum = 0.0;
            for (int j = 0; j < cases.Count; j++)
            {
                sum += i[j] * cases[j].LogOdds;
            }
            return sum;
        }
    }
}
