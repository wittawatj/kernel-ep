/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.StringCapitalized(int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "StringCapitalized", typeof(int))]
    [Quality(QualityBand.Experimental)]
    public static class StringCapitalizedOfMinLengthOp
    {
        #region EP messages

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="minLength">Constant value for <c>minLength</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>str</c> conditioned on the given values.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(int minLength)
        {
            return StringDistribution.Capitalized(minLength);
        }

        #endregion
    }
}
