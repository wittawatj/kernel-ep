// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// Exception which is thrown when a constraint is violated.  This
    /// occurs when an observation does not hold true or a weight is 0.
    /// </summary>
    public class ConstraintViolatedException : Exception
    {
        /// <summary>
        /// Construct a constraint violated exception with a specified error message 
        /// </summary>
        /// <param name="s"></param>
        public ConstraintViolatedException(string s)
            : base(s)
        {
        }
    }

    /// <summary>
    /// A repository of commonly used constraint methods.
    /// </summary>
    public static class Constrain
    {
        /// <summary>
        /// Constrains a value to be equal to a sample from dist.
        /// </summary>
        /// <typeparam name="TDomain">Domain type</typeparam>
        /// <typeparam name="TDistribution">Distribution type</typeparam>
        /// <param name="value">Value</param>
        /// <param name="dist">Distribution instance</param>
        [Stochastic]
        public static void EqualRandom<TDomain, TDistribution>(TDomain value, TDistribution dist)
            where TDistribution : Sampleable<TDomain>
        {
            if (!value.Equals(dist.Sample()))
                throw new ConstraintViolatedException(value + " != " + dist);
        }

        /// <summary>
        /// Constrains a value to be equal to another value.
        /// </summary>
        /// <typeparam name="T">Value type</typeparam>
        /// <param name="A">First value</param>
        /// <param name="B">Second value</param>
        public static void Equal<T>(T A, T B)
        {
            if (!A.Equals(B))
                throw new ConstraintViolatedException(A + " != " + B);
        }

        /// <summary>
        /// Constrains a set of integers to contain a particular integer.
        /// </summary>
        /// <param name="set">The set of integers, specified as a list</param>
        /// <param name="i">The integer which the set must contain</param>
        [Hidden]
        public static void Contain(IList<int> set, int i)
        {
            if (!set.Contains(i))
            {
                throw new ConstraintViolatedException(
                    "Containment constraint violated (the supplied set did not contain the integer " + i + ")");
            }
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Constrain.EqualRandom{TDomain, TDistribution}(TDomain, TDistribution)" />, given random arguments to the function.</summary>
    /// <typeparam name="TDomain">The domain of the constrained variables.</typeparam>
    [FactorMethod(typeof(Constrain), "EqualRandom<,>")]
    [Quality(QualityBand.Mature)]
    public static class ConstrainEqualRandomOp<TDomain>    
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>The outgoing EP message to the <c>value</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(dist) p(dist) factor(value,dist)]/p(value)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution ValueAverageConditional<TDistribution>([IsReturned] TDistribution dist)
        {
            return dist;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(value,dist) p(value,dist) factor(value,dist))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDistribution value, TDistribution dist)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return value.GetLogAverageOf(dist);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(value,dist) p(value,dist) factor(value,dist))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="dist" /> is not a proper distribution.</exception>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDomain value, [Proper] TDistribution dist)
            where TDistribution : CanGetLogProb<TDomain>
        {
            return dist.GetLogProb(value);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(value,dist) p(value,dist) factor(value,dist))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDistribution value, TDistribution dist)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return LogAverageFactor(value, dist);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(value,dist) p(value,dist) factor(value,dist))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDomain value, TDistribution dist)
            where TDistribution : CanGetLogProb<TDomain>
        {
            return LogAverageFactor(value, dist);
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(value,dist) p(value,dist) log(factor(value,dist))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double AverageLogFactor<TDistribution>(TDistribution value, TDistribution dist)
            where TDistribution : CanGetAverageLog<TDistribution>
        {
            return value.GetAverageLog(dist);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(value,dist) p(value,dist) log(factor(value,dist))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="dist" /> is not a proper distribution.</exception>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double AverageLogFactor<TDistribution>(TDomain value, [Proper] TDistribution dist)
            where TDistribution : CanGetLogProb<TDomain>
        {
            return dist.GetLogProb(value);
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>The outgoing VMP message to the <c>value</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>value</c>. The formula is <c>exp(sum_(dist) p(dist) log(factor(value,dist)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution ValueAverageLogarithm<TDistribution>([IsReturned] TDistribution dist)
        {
            return dist;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Constrain.Equal{T}(T, T)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the constrained variables.</typeparam>
    [FactorMethod(typeof(Constrain), "Equal<>")]
    [Quality(QualityBand.Mature)]
    public static class ConstrainEqualOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDistribution a, TDistribution b)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return a.GetLogAverageOf(b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(T a, TDistribution b)
            where TDistribution : CanGetLogProb<T>
        {
            return b.GetLogProb(a);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDistribution a, T b)
            where TDistribution : CanGetLogProb<T>
        {
            return a.GetLogProb(b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(T a, T b)
        {
            IEqualityComparer<T> equalityComparer = Utils.Util.GetEqualityComparer<T>();
            return (equalityComparer.Equals(a, b) ? 0.0 : Double.NegativeInfinity);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDistribution a, TDistribution b)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return LogAverageFactor(a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(T a, TDistribution b)
            where TDistribution : CanGetLogProb<T>
        {
            return LogAverageFactor(a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDistribution a, T b)
            where TDistribution : CanGetLogProb<T>
        {
            return LogAverageFactor(a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="a">Incoming message from <c>A</c>.</param>
        /// <param name="b">Incoming message from <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(A,B) p(A,B) factor(A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(T a, T b)
        {
            return LogAverageFactor(a, b);
        }

        /// <summary>EP message to <c>A</c>.</summary>
        /// <param name="B">Incoming message from <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>A</c> as the random arguments are varied. The formula is <c>proj[p(A) sum_(B) p(B) factor(A,B)]/p(A)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AAverageConditional<TDistribution>([IsReturned] TDistribution B, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(B);
            return result;
        }

        /// <summary>EP message to <c>A</c>.</summary>
        /// <param name="B">Incoming message from <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>A</c> as the random arguments are varied. The formula is <c>proj[p(A) sum_(B) p(B) factor(A,B)]/p(A)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AAverageConditional<TDistribution>(T B, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            result.Point = B;
            return result;
        }

        /// <summary>EP message to <c>B</c>.</summary>
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>B</c> as the random arguments are varied. The formula is <c>proj[p(B) sum_(A) p(A) factor(A,B)]/p(B)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BAverageConditional<TDistribution>([IsReturned] TDistribution A, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(A);
            return result;
        }

        /// <summary>EP message to <c>B</c>.</summary>
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>B</c> as the random arguments are varied. The formula is <c>proj[p(B) sum_(A) p(A) factor(A,B)]/p(B)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BAverageConditional<TDistribution>(T A, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            return AAverageConditional<TDistribution>(A, result);
        }

        //-- VMP -----------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        private const string NotSupportedMessage = "VMP does not support Constrain.Equal between random variables";

        /// <summary>VMP message to <c>A</c>.</summary>
        /// <param name="B">Incoming message from <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>A</c>. The formula is <c>exp(sum_(B) p(B) log(factor(A,B)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        [NotSupported(NotSupportedMessage)]
        public static TDistribution AAverageLogarithm<TDistribution>(TDistribution B, TDistribution result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>B</c>.</summary>
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>B</c>. The formula is <c>exp(sum_(A) p(A) log(factor(A,B)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        [NotSupported(NotSupportedMessage)]
        public static TDistribution BAverageLogarithm<TDistribution>(TDistribution A, TDistribution result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>A</c>.</summary>
        /// <param name="B">Incoming message from <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>A</c>. The formula is <c>exp(sum_(B) p(B) log(factor(A,B)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AAverageLogarithm<TDistribution>(T B, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            result.Point = B;
            return result;
        }

        /// <summary>VMP message to <c>B</c>.</summary>
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>B</c>. The formula is <c>exp(sum_(A) p(A) log(factor(A,B)))</c>.</para>
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BAverageLogarithm<TDistribution>(T A, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            return AAverageLogarithm<TDistribution>(A, result);
        }

        //-- Max product ----------------------------------------------------------------------

        /// <summary />
        /// <param name="B">Incoming message from <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AMaxConditional<TDistribution>([IsReturned] TDistribution B, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(B);
            return result;
        }

        /// <summary />
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BMaxConditional<TDistribution>([IsReturned] TDistribution A, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(A);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Constrain.Contain(IList{int}, int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Constrain), "Contain")]
    [Quality(QualityBand.Experimental)]
    public class ConstrainContainOp
    {
        /// <summary>VMP message to <c>set</c>.</summary>
        /// <param name="i">Constant value for <c>i</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>set</c> conditioned on the given values.</para>
        /// </remarks>
        public static BernoulliIntegerSubset SetAverageLogarithm(int i, BernoulliIntegerSubset result)
        {
            result.SetToUniform();
            result.SparseBernoulliList[i] = Bernoulli.PointMass(true);
            return result;
        }
    }
}
