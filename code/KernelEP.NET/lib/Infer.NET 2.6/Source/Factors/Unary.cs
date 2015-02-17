// (C) Copyright 2008 Microsoft Research Cambridge

using MicrosoftResearch.Infer.Distributions;

[assembly: MicrosoftResearch.Infer.Factors.HasMessageFunctions]

namespace MicrosoftResearch.Infer.Factors
{
    /// <summary>Provides outgoing messages for <see cref="Factor.Random{DomainType}(Sampleable{DomainType})" />, given random arguments to the function.</summary>
    /// <typeparam name="DomainType">The type of the sampled variable.</typeparam>
    [FactorMethod(typeof(Factor), "Random<>")]
    [Quality(QualityBand.Mature)]
    public static class UnaryOp<DomainType>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="random">Incoming message from <c>random</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(random,dist) p(random,dist) factor(random,dist))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double LogAverageFactor<T>(T random, T dist)
            where T : CanGetLogAverageOf<T>
        {
            return dist.GetLogAverageOf(random);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="random">Incoming message from <c>random</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(random,dist) p(random,dist) factor(random,dist))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double LogAverageFactor<T>(DomainType random, T dist)
            where T : CanGetLogProb<DomainType>
        {
            return dist.GetLogProb(random);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="random">Incoming message from <c>random</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(random,dist) p(random,dist) factor(random,dist) / sum_random p(random) messageTo(random))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<T>(T random, T dist)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="random">Incoming message from <c>random</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(random,dist) p(random,dist) factor(random,dist) / sum_random p(random) messageTo(random))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double LogEvidenceRatio<T>(DomainType random, T dist)
            where T : CanGetLogProb<DomainType>
        {
            return LogAverageFactor(random, dist);
        }

        /// <summary>EP message to <c>random</c>.</summary>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>The outgoing EP message to the <c>random</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>random</c> as the random arguments are varied. The formula is <c>proj[p(random) sum_(dist) p(dist) factor(random,dist)]/p(random)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static T RandomAverageConditional<T>([IsReturned] T dist)
        {
            return dist;
        }

        //-- VMP ---------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="random">Incoming message from <c>random</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(random,dist) p(random,dist) log(factor(random,dist))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double AverageLogFactor<T>(T random, T dist)
            where T : CanGetAverageLog<T>
        {
            return random.GetAverageLog(dist);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="random">Incoming message from <c>random</c>.</param>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(random,dist) p(random,dist) log(factor(random,dist))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double AverageLogFactor<T>(DomainType random, T dist)
            where T : CanGetLogProb<DomainType>
        {
            return dist.GetLogProb(random);
        }

        /// <summary>VMP message to <c>random</c>.</summary>
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns>The outgoing VMP message to the <c>random</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>random</c>. The formula is <c>exp(sum_(dist) p(dist) log(factor(random,dist)))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static T RandomAverageLogarithm<T>([IsReturned] T dist)
        {
            return dist;
        }

        //-- Max product ---------------------------------------------------------------------------------------

        /// <summary />
        /// <param name="dist">Incoming message from <c>dist</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Bernoulli RandomMaxConditional([IsReturned] Bernoulli dist)
        {
            return dist;
        }

        /// <summary />
        /// <param name="dist">Incoming message from <c>dist</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="dist" /> is not a proper distribution.</exception>
        public static UnnormalizedDiscrete RandomMaxConditional([SkipIfUniform] Discrete dist)
        {
            return UnnormalizedDiscrete.FromDiscrete(dist);
        }
    }
}
