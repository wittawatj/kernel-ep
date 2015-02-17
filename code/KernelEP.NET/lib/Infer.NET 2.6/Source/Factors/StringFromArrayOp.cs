/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.StringFromArray(Char[])" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "StringFromArray")]
    [Quality(QualityBand.Experimental)]
    public static class StringFromArrayOp
    {
        #region EP messages

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="characters">Incoming message from <c>characters</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(characters) p(characters) factor(str,characters)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(IList<DiscreteChar> characters)
        {
            StringDistribution result = StringDistribution.Empty();
            for (int i = 0; i < characters.Count; ++i)
            {
                result.AppendInPlace(characters[i]);
            }

            return result;
        }

        /// <summary>EP message to <c>characters</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="characters">Incoming message from <c>characters</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>characters</c> as the random arguments are varied. The formula is <c>proj[p(characters) sum_(str) p(str) factor(str,characters)]/p(characters)</c>.</para>
        /// </remarks>
        /// <typeparam name="TDiscreteCharList">The type of an outgoing message to <c>chars</c>.</typeparam>
        public static TDiscreteCharList CharactersAverageConditional<TDiscreteCharList>(
            StringDistribution str, IList<DiscreteChar> characters, TDiscreteCharList result)
            where TDiscreteCharList : IList<DiscreteChar>
        {
            for (int i = 0; i < characters.Count; ++i)
            {
                // TODO: perhaps there is a faster way to extract the distribution of interest
                var reweightedStr = str.Product(GetCharWeighter(characters, i));
                var outgoingMessageAsStr = SubstringOp.SubAverageConditional(reweightedStr, i, 1);
                if (outgoingMessageAsStr.IsZero())
                {
                    throw new AllZeroException("Impossible model detected in StringFromCharsOp.");
                }
                
                result[i] = SingleOp.CharacterAverageConditional(outgoingMessageAsStr);
            }

            return result;
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="characters">Incoming message from <c>characters</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(characters,str) p(characters,str) factor(str,characters) / sum_str p(str) messageTo(str))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(IList<DiscreteChar> characters, StringDistribution str)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="characters">Incoming message from <c>characters</c>.</param>
        /// <param name="str">Constant value for <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(characters) p(characters) factor(str,characters))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(IList<DiscreteChar> characters, string str)
        {
            StringDistribution toStr = StrAverageConditional(characters);
            return toStr.GetLogProb(str);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Creates a string distribution <c>P(s) = \prod_i P_i(s_i)^I[i != j]</c>,
        /// where <c>P_i(c)</c> is a given array of character distributions and <c>j</c> is a given position in the array.
        /// </summary>
        /// <param name="characters">The distributions over individual characters.</param>
        /// <param name="excludedPos">The character to skip.</param>
        /// <returns>The created distribution.</returns>
        private static StringDistribution GetCharWeighter(IList<DiscreteChar> characters, int excludedPos)
        {
            StringDistribution result = StringDistribution.Empty();
            for (int i = 0; i < characters.Count; ++i)
            {
                result.AppendInPlace(i == excludedPos ? DiscreteChar.Uniform() : characters[i]);
            }

            return result;
        }

        #endregion
    }
}
