/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.Char(Vector)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Char")]
    [Quality(QualityBand.Experimental)]
    public static class CharFromProbabilitiesOp
    {
        #region EP messages

        /// <summary>EP message to <c>character</c>.</summary>
        /// <param name="probabilities">Incoming message from <c>probabilities</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>character</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>character</c> as the random arguments are varied. The formula is <c>proj[p(character) sum_(probabilities) p(probabilities) factor(character,probabilities)]/p(character)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probabilities" /> is not a proper distribution.</exception>
        public static DiscreteChar CharacterAverageConditional([SkipIfUniform] Dirichlet probabilities)
        {
            Discrete resultAsDiscrete = Discrete.Uniform(probabilities.Dimension, probabilities.Sparsity);
            DiscreteFromDirichletOp.SampleAverageConditional(probabilities, resultAsDiscrete);
            return DiscreteChar.FromVector(resultAsDiscrete.GetProbs());
        }

        /// <summary>EP message to <c>probabilities</c>.</summary>
        /// <param name="character">Incoming message from <c>character</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="probabilities">Incoming message from <c>probabilities</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>probabilities</c> as the random arguments are varied. The formula is <c>proj[p(probabilities) sum_(character) p(character) factor(character,probabilities)]/p(probabilities)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="character" /> is not a proper distribution.</exception>
        public static Dirichlet ProbabilitiesAverageConditional([SkipIfUniform] DiscreteChar character, Dirichlet probabilities, Dirichlet result)
        {
            return DiscreteFromDirichletOp.ProbsAverageConditional(character.GetInternalDiscrete(), probabilities, result);
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="probabilities">Incoming message from <c>probabilities</c>.</param>
        /// <param name="character">Incoming message from <c>character</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(probabilities,character) p(probabilities,character) factor(character,probabilities) / sum_character p(character) messageTo(character))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Dirichlet probabilities, DiscreteChar character)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="probabilities">Incoming message from <c>probabilities</c>.</param>
        /// <param name="character">Constant value for <c>character</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(probabilities) p(probabilities) factor(character,probabilities))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Dirichlet probabilities, char character)
        {
            return DiscreteFromDirichletOp.LogEvidenceRatio(character, probabilities);
        }

        #endregion
    }
}
