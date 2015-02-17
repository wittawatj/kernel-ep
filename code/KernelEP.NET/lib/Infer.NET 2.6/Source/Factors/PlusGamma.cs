namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.Plus(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Experimental)]
    public static class PlusGammaVmpOp
    {
        /// <summary>VMP message to <c>Sum</c>.</summary>
        /// <param name="a">Incoming message from <c>A</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="b">Incoming message from <c>B</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>Sum</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Sum</c> as the random arguments are varied. The formula is <c>proj[sum_(A,B) p(A,B) factor(Sum,A,B)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="a" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="b" /> is not a proper distribution.</exception>
        public static Gamma SumAverageLogarithm([Proper] Gamma a, [Proper] Gamma b)
        {
            // return a Gamma with the correct moments
            double ma, va, mb, vb;
            a.GetMeanAndVariance(out ma, out va);
            b.GetMeanAndVariance(out mb, out vb);
            return Gamma.FromMeanAndVariance(ma + mb, va + vb);
        }

        /// <summary>VMP message to <c>A</c>.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="a">Incoming message from <c>A</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="b">Incoming message from <c>B</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>A</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>A</c>. Because the factor is deterministic, <c>Sum</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(B) p(B) log(sum_Sum p(Sum) factor(Sum,A,B)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="a" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="b" /> is not a proper distribution.</exception>
        public static Gamma AAverageLogarithm([SkipIfUniform] Gamma sum, [Proper] Gamma a, [Proper] Gamma b)
        {
            // f = int_sum sum^(ss-1)*exp(-sr*sum)*delta(sum = a+b) dsum
            //   = (a+b)^(ss-1)*exp(-sr*(a+b))
            // log(f) = (ss-1)*log(a+b) - sr*(a+b)
            // apply a lower bound:
            // log(a+b) >= q*log(a) + (1-q)*log(b) - q*log(q) - (1-q)*log(1-q)
            // optimal q = exp(a)/(exp(a)+exp(b)) if (a,b) are fixed
            // optimal q = exp(E[log(a)])/(exp(E[log(a)])+exp(E[log(b)]))  if (a,b) are random
            // This generalizes the bound used by Cemgil (2008).
            // The message to A has shape (ss-1)*q + 1 and rate sr.
            double x = sum.Shape - 1;
            double ma = Math.Exp(a.GetMeanLog());
            double mb = Math.Exp(b.GetMeanLog());
            double p = ma / (ma + mb);
            double m = x * p;
            return Gamma.FromShapeAndRate(m + 1, sum.Rate);
        }

        /// <summary>VMP message to <c>B</c>.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="a">Incoming message from <c>A</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="b">Incoming message from <c>B</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>B</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>B</c>. Because the factor is deterministic, <c>Sum</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(A) p(A) log(sum_Sum p(Sum) factor(Sum,A,B)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="a" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="b" /> is not a proper distribution.</exception>
        public static Gamma BAverageLogarithm([SkipIfUniform] Gamma sum, [Proper] Gamma a, [Proper] Gamma b)
        {
            return AAverageLogarithm(sum, b, a);
        }
    }
}
