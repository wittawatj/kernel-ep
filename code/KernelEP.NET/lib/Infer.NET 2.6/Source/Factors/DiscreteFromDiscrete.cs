// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.Discrete(int, Matrix)" />, given random arguments to the function.</summary>
    [FactorMethod(new String[] { "sample", "selector", "probs" }, typeof(Factor), "Discrete", typeof(int), typeof(Matrix))]
    [Quality(QualityBand.Experimental)]
    public static class DiscreteFromDiscreteOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="probs">Constant value for <c>probs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,selector) p(sample,selector) factor(sample,selector,probs))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Discrete sample, Discrete selector, Matrix probs)
        {
            return Math.Log(probs.QuadraticForm(selector.GetProbs(), sample.GetProbs()));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="probs">Constant value for <c>probs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,selector) p(sample,selector) factor(sample,selector,probs) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Discrete sample, Discrete selector, Matrix probs)
        {
            // use this if the rows are not normalized
            Discrete toSample = SampleAverageConditional(selector, probs, Discrete.Uniform(sample.Dimension, sample.Sparsity));
            return LogAverageFactor(sample, selector, probs)
                   - toSample.GetLogAverageOf(sample);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="selector">Incoming message from <c>selector</c>.</param>
        /// <param name="probs">Constant value for <c>probs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(selector) p(selector) factor(sample,selector,probs)]/p(sample)</c>.</para>
        /// </remarks>
        public static Discrete SampleAverageConditional(Discrete selector, Matrix probs, Discrete result)
        {
            Vector v = result.GetWorkspace();
            v.SetToProduct(selector.GetProbs(), probs);
            result.SetProbs(v);
            return result;
        }

        /// <summary>EP message to <c>selector</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="probs">Constant value for <c>probs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>selector</c> as the random arguments are varied. The formula is <c>proj[p(selector) sum_(sample) p(sample) factor(sample,selector,probs)]/p(selector)</c>.</para>
        /// </remarks>
        public static Discrete SelectorAverageConditional(Discrete sample, Matrix probs, Discrete result)
        {
            Vector v = result.GetWorkspace();
            v.SetToProduct(probs, sample.GetProbs());
            result.SetProbs(v);
            return result;
        }
    }
}
