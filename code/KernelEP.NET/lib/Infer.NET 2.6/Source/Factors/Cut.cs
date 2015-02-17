// (C) Cutright 2013 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.Cut{T}(T)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable being copied.</typeparam>
    [FactorMethod(typeof(Factor), "Cut<>")]
    [Quality(QualityBand.Preview)]
    public static class CutOp<T>
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>value</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static TDist ValueAverageConditional<TDist>(TDist result)
            where TDist : IDistribution<T>
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>EP message to <c>cut</c>.</summary>
        /// <param name="Value">Incoming message from <c>value</c>.</param>
        /// <returns>The outgoing EP message to the <c>cut</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>cut</c> as the random arguments are varied. The formula is <c>proj[p(cut) sum_(value) p(value) factor(cut,value)]/p(cut)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist CutAverageConditional<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        /// <summary>EP message to <c>cut</c>.</summary>
        /// <param name="Value">Incoming message from <c>value</c>.</param>
        /// <returns>The outgoing EP message to the <c>cut</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>cut</c> as the random arguments are varied. The formula is <c>proj[p(cut) sum_(value) p(value) factor(cut,value)]/p(cut)</c>.</para>
        /// </remarks>
        public static T CutAverageConditional([IsReturned] T Value)
        {
            return Value;
        }
    }
}
