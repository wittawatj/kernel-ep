// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.BernoulliFromDiscrete(int, double[])" />, given random arguments to the function.</summary>
    /// <exclude/>
    [FactorMethod(new string[] { "sample", "index", "probTrue" }, typeof(Factor), "BernoulliFromDiscrete")]
    public static class BernoulliFromDiscreteOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="index">Incoming message from <c>index</c>.</param>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(index) p(index) factor(sample,index,probTrue))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool sample, Discrete index, double[] probTrue)
        {
            double p = 0;
            for (int i = 0; i < index.Dimension; i++)
            {
                p += probTrue[i] * index[i];
            }
            if (!sample)
                p = 1 - p;
            return Math.Log(p);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="index">Incoming message from <c>index</c>.</param>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(index) p(index) factor(sample,index,probTrue))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool sample, Discrete index, double[] probTrue)
        {
            return LogAverageFactor(sample, index, probTrue);
        }

        /// <summary>Gibbs message to <c>sample</c>.</summary>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>The outgoing Gibbs message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli SampleConditional(int index, double[] ProbTrue)
        {
            Bernoulli result = new Bernoulli();
            result.SetProbTrue(ProbTrue[index]);
            return result;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli SampleAverageConditional(int index, double[] ProbTrue)
        {
            return SampleConditional(index, ProbTrue);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="index">Constant value for <c>index</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli SampleAverageLogarithm(int index, double[] ProbTrue)
        {
            return SampleConditional(index, ProbTrue);
        }

        /// <summary>Gibbs message to <c>index</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>index</c> conditioned on the given values.</para>
        /// </remarks>
        public static Discrete IndexConditional(bool sample, double[] ProbTrue, Discrete result)
        {
            if (result == default(Discrete))
                result = Discrete.Uniform(ProbTrue.Length);
            Vector prob = result.GetWorkspace();
            if (sample)
            {
                prob.SetTo(ProbTrue);
            }
            else
            {
                prob.SetTo(ProbTrue);
                prob.SetToDifference(1.0, prob);
            }
            result.SetProbs(prob);
            return result;
        }

        /// <summary>EP message to <c>index</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>index</c> conditioned on the given values.</para>
        /// </remarks>
        public static Discrete IndexAverageConditional(bool sample, double[] ProbTrue, Discrete result)
        {
            return IndexConditional(sample, ProbTrue, result);
        }

        /// <summary>VMP message to <c>index</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>index</c> conditioned on the given values.</para>
        /// </remarks>
        public static Discrete IndexAverageLogarithm(bool sample, double[] ProbTrue, Discrete result)
        {
            return IndexConditional(sample, ProbTrue, result);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="index">Incoming message from <c>index</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(index) p(index) factor(sample,index,probTrue)]/p(sample)</c>.</para>
        /// </remarks>
        public static Bernoulli SampleAverageConditional(Discrete index, double[] ProbTrue)
        {
            Bernoulli result = new Bernoulli();
            // E[X] = sum_Y p(Y) ProbTrue[Y]
            double p = 0;
            for (int i = 0; i < index.Dimension; i++)
            {
                p += ProbTrue[i] * index[i];
            }
            result.SetProbTrue(p);
            return result;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="index">Incoming message from <c>index</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(index) p(index) log(factor(sample,index,probTrue)))</c>.</para>
        /// </remarks>
        public static Bernoulli SampleAverageLogarithm(Discrete index, double[] ProbTrue)
        {
            Bernoulli result = new Bernoulli();
            // E[sum_k I(Y=k) (X*log(ProbTrue[k]) + (1-X)*log(1-ProbTrue[k]))]
            // = X*(sum_k p(Y=k) log(ProbTrue[k])) + (1-X)*(sum_k p(Y=k) log(1-ProbTrue[k]))
            // p(X=true) =propto prod_k ProbTrue[k]^p(Y=k)
            // log(p(X=true)/p(X=false)) = sum_k p(Y=k) log(ProbTrue[k]/(1-ProbTrue[k]))
            double s = 0;
            for (int i = 0; i < index.Dimension; i++)
            {
                s += index[i] * MMath.Logit(ProbTrue[i]);
            }
            result.LogOdds = s;
            return result;
        }

        /// <summary>EP message to <c>index</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>index</c> as the random arguments are varied. The formula is <c>proj[p(index) sum_(sample) p(sample) factor(sample,index,probTrue)]/p(index)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static Discrete IndexAverageConditional([SkipIfUniform] Bernoulli sample, double[] ProbTrue, Discrete result)
        {
            if (result == default(Discrete))
                result = Discrete.Uniform(ProbTrue.Length);
            // p(Y) = ProbTrue[Y]*p(X=true) + (1-ProbTrue[Y])*p(X=false)
            Vector probs = result.GetWorkspace();
            double p = sample.GetProbTrue();
            probs.SetTo(ProbTrue);
            probs.SetToProduct(probs, 2.0 * p - 1.0);
            probs.SetToSum(probs, 1.0 - p);
            result.SetProbs(probs);
            return result;
        }

        /// <summary>VMP message to <c>index</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="ProbTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>index</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,index,probTrue)))</c>.</para>
        /// </remarks>
        public static Discrete IndexAverageLogarithm(Bernoulli sample, double[] ProbTrue, Discrete result)
        {
            if (result == default(Discrete))
                result = Discrete.Uniform(ProbTrue.Length);
            // E[sum_k I(Y=k) (X*log(ProbTrue[k]) + (1-X)*log(1-ProbTrue[k]))]
            // = sum_k I(Y=k) (p(X=true)*log(ProbTrue[k]) + p(X=false)*log(1-ProbTrue[k]))
            // p(Y=k) =propto ProbTrue[k]^p(X=true) (1-ProbTrue[k])^p(X=false)
            Vector probs = result.GetWorkspace();
            double p = sample.GetProbTrue();
            probs.SetTo(ProbTrue);
            probs.SetToFunction(probs, x => Math.Pow(x, p) * Math.Pow(1.0 - x, 1.0 - p));
            result.SetProbs(probs);
            return result;
        }
    }
}
