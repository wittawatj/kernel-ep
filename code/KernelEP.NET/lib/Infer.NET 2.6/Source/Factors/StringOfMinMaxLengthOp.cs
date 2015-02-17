/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.String(int, int, DiscreteChar)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "String", typeof(int), typeof(int), typeof(DiscreteChar))]
    [Quality(QualityBand.Experimental)]
    public static class StringOfMinMaxLengthOp
    {
        #region EP messages

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="allowedChars">Constant value for <c>allowedChars</c>.</param>
        /// <param name="minLength">Constant value for <c>minLength</c>.</param>
        /// <param name="maxLength">Constant value for <c>maxLength</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>str</c> conditioned on the given values.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(DiscreteChar allowedChars, int minLength, int maxLength)
        {
            return StringDistribution.Repeat(allowedChars, minLength, maxLength);
        }

        #endregion
    }
}
