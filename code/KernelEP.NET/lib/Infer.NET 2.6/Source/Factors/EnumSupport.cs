// (C) Copyright 2008 Microsoft Research Cambridge

using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MicrosoftResearch.Infer.Factors
{
    /// <summary>
    /// Provides factors and operators for using Enum types.
    /// </summary>
    public class EnumSupport
    {
        /// <summary>
        /// Converts an Enum to an Int
        /// </summary>
        /// <param name="en"></param>
        /// <returns></returns>
        [ParameterNames("Int", "Enum")]
        public static int EnumToInt<TEnum>(TEnum en)
        {
            return (int)(object)en;
        }

        /// <summary>
        /// Samples an enum value from a discrete enum distribution.
        /// </summary>
        /// <typeparam name="TEnum">The type of the enum to sample</typeparam>
        /// <param name="probs">Vector of the probability of each Enum value, in order</param>
        /// <returns>An enum sampled from the distribution</returns>
        [Stochastic]
        [ParameterNames("Sample", "Probs")]
        public static TEnum DiscreteEnum<TEnum>(Vector probs)
        {
            int i = MicrosoftResearch.Infer.Distributions.Discrete.Sample(probs);
            return (TEnum)Enum.GetValues(typeof(TEnum)).GetValue(i);
        }

        /// <summary>
        /// Test if two enums are equal.
        /// </summary>
        /// <param name="a">First integer</param>
        /// <param name="b">Second integer</param>
        /// <returns>True if a==b.</returns>
        public static bool AreEqual<TEnum>(TEnum a, TEnum b)
        {
            return EnumToInt<TEnum>(a) == EnumToInt<TEnum>(b);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="EnumSupport.EnumToInt{TEnum}(TEnum)" />, given random arguments to the function.</summary>
    /// <typeparam name="TEnum">The type of the enumeration.</typeparam>
    [FactorMethod(typeof(EnumSupport), "EnumToInt<>")]
    [Quality(QualityBand.Preview)]
    public static class EnumToIntOp<TEnum>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="Int">Constant value for <c>Int</c>.</param>
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Enum) p(Enum) factor(Int,Enum))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(int Int, TEnum Enum)
        {
            return (EnumSupport.EnumToInt(Enum) == Int) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Int">Constant value for <c>Int</c>.</param>
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Enum) p(Enum) factor(Int,Enum))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int Int, TEnum Enum)
        {
            return LogAverageFactor(Int, Enum);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="Int">Constant value for <c>Int</c>.</param>
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(int Int, TEnum Enum)
        {
            return LogAverageFactor(Int, Enum);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Int">Constant value for <c>Int</c>.</param>
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Enum) p(Enum) factor(Int,Enum))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(int Int, DiscreteEnum<TEnum> Enum)
        {
            return Enum.GetLogProb((TEnum)(object)Int);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Int">Constant value for <c>Int</c>.</param>
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Enum) p(Enum) factor(Int,Enum))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int Int, DiscreteEnum<TEnum> Enum)
        {
            return LogAverageFactor(Int, Enum);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Int">Incoming message from <c>Int</c>.</param>
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Int,Enum) p(Int,Enum) factor(Int,Enum))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Discrete Int, TEnum Enum)
        {
            return Int.GetLogProb((int)(object)Enum);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Int">Incoming message from <c>Int</c>.</param>
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <param name="to_Int">Outgoing message to <c>Int</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Int,Enum) p(Int,Enum) factor(Int,Enum))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Discrete Int, DiscreteEnum<TEnum> Enum, [Fresh] Discrete to_Int)
        {
            return to_Int.GetLogAverageOf(Int);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Int">Incoming message from <c>Int</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Int) p(Int) factor(Int,Enum) / sum_Int p(Int) messageTo(Int))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Discrete Int)
        {
            return 0.0;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Int,Enum))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>EP message to <c>Int</c>.</summary>
        /// <param name="Enum">Incoming message from <c>Enum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Int</c> as the random arguments are varied. The formula is <c>proj[p(Int) sum_(Enum) p(Enum) factor(Int,Enum)]/p(Int)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Enum" /> is not a proper distribution.</exception>
        public static Discrete IntAverageConditional([SkipIfUniform] DiscreteEnum<TEnum> Enum, Discrete result)
        {
            result.SetProbs(Enum.GetWorkspace());
            return result;
        }

        /// <summary />
        /// <param name="Enum">Incoming message from <c>Enum</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Discrete IntAverageConditionalInit([IgnoreDependency] DiscreteEnum<TEnum> Enum)
        {
            return Discrete.Uniform(Enum.Dimension);
        }

        /// <summary>EP message to <c>Enum</c>.</summary>
        /// <param name="Int">Incoming message from <c>Int</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Enum</c> as the random arguments are varied. The formula is <c>proj[p(Enum) sum_(Int) p(Int) factor(Int,Enum)]/p(Enum)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Int" /> is not a proper distribution.</exception>
        public static DiscreteEnum<TEnum> EnumAverageConditional([SkipIfUniform] Discrete Int, DiscreteEnum<TEnum> result)
        {
            result.SetProbs(Int.GetWorkspace());
            return result;
        }

        /// <summary>EP message to <c>Enum</c>.</summary>
        /// <param name="Int">Constant value for <c>Int</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Enum</c> conditioned on the given values.</para>
        /// </remarks>
        public static DiscreteEnum<TEnum> EnumAverageConditional(int Int, DiscreteEnum<TEnum> result)
        {
            result.Point = (TEnum)Enum.GetValues(typeof(TEnum)).GetValue(Int);
            return result;
        }

        /// <summary>VMP message to <c>Int</c>.</summary>
        /// <param name="Enum">Incoming message from <c>Enum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Int</c> as the random arguments are varied. The formula is <c>proj[sum_(Enum) p(Enum) factor(Int,Enum)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Enum" /> is not a proper distribution.</exception>
        public static Discrete IntAverageLogarithm([SkipIfUniform] DiscreteEnum<TEnum> Enum, Discrete result)
        {
            return IntAverageConditional(Enum, result);
        }

        /// <summary>VMP message to <c>Enum</c>.</summary>
        /// <param name="Int">Incoming message from <c>Int</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Enum</c> with <c>Int</c> integrated out. The formula is <c>sum_Int p(Int) factor(Int,Enum)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Int" /> is not a proper distribution.</exception>
        public static DiscreteEnum<TEnum> EnumAverageLogarithm([SkipIfUniform] Discrete Int, DiscreteEnum<TEnum> result)
        {
            return EnumAverageConditional(Int, result);
        }

        /// <summary>VMP message to <c>Enum</c>.</summary>
        /// <param name="Int">Constant value for <c>Int</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Enum</c> conditioned on the given values.</para>
        /// </remarks>
        public static DiscreteEnum<TEnum> EnumAverageLogarithm(int Int, DiscreteEnum<TEnum> result)
        {
            return EnumAverageConditional(Int, result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="EnumSupport.DiscreteEnum{TEnum}(Vector)" />, given random arguments to the function.</summary>
    /// <typeparam name="TEnum">The type of the enumeration.</typeparam>
    /// <remarks>
    /// This class provides operators which have <see cref="Enum"/> arguments.  
    /// The rest are provided by <see cref="DiscreteFromDirichletOp"/>.
    /// </remarks>
    [FactorMethod(typeof(EnumSupport), "DiscreteEnum<>")]
    [Quality(QualityBand.Stable)]
    public static class DiscreteEnumFromDirichletOp<TEnum>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample,Probs) p(Sample,Probs) factor(Sample,Probs))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(TEnum sample, Dirichlet probs)
        {
            return DiscreteFromDirichletOp.LogAverageFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample) p(Sample) factor(Sample,Probs))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(TEnum sample, Vector probs)
        {
            return DiscreteFromDirichletOp.LogAverageFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(Sample,Probs) p(Sample,Probs) log(factor(Sample,Probs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(TEnum sample, Dirichlet probs)
        {
            return DiscreteFromDirichletOp.AverageLogFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(Sample) p(Sample) log(factor(Sample,Probs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(TEnum sample, Vector probs)
        {
            return DiscreteFromDirichletOp.AverageLogFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample,Probs) p(Sample,Probs) factor(Sample,Probs) / sum_Sample p(Sample) messageTo(Sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(TEnum sample, Dirichlet probs)
        {
            return DiscreteFromDirichletOp.LogEvidenceRatio(EnumSupport.EnumToInt(sample), probs);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample) p(Sample) factor(Sample,Probs) / sum_Sample p(Sample) messageTo(Sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(TEnum sample, Vector probs)
        {
            return DiscreteFromDirichletOp.LogEvidenceRatio(EnumSupport.EnumToInt(sample), probs);
        }

        /// <summary>EP message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Probs</c> as the random arguments are varied. The formula is <c>proj[p(Probs) sum_(Sample) p(Sample) factor(Sample,Probs)]/p(Probs)</c>.</para>
        /// </remarks>
        public static Dirichlet ProbsAverageConditional(TEnum sample, Dirichlet result)
        {
            return DiscreteFromDirichletOp.ProbsAverageConditional(EnumSupport.EnumToInt(sample), result);
        }

        /// <summary>VMP message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Probs</c>. The formula is <c>exp(sum_(Sample) p(Sample) log(factor(Sample,Probs)))</c>.</para>
        /// </remarks>
        public static Dirichlet ProbsAverageLogarithm(TEnum sample, Dirichlet result)
        {
            return DiscreteFromDirichletOp.ProbsAverageLogarithm(EnumSupport.EnumToInt(sample), result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="EnumSupport.AreEqual{TEnum}(TEnum, TEnum)" />, given random arguments to the function.</summary>
    /// <typeparam name="TEnum">The type of the enumeration.</typeparam>
    /// <remarks>
    /// This class only implements enumeration-specific overloads that are not provided by <see cref="DiscreteAreEqualOp"/>.
    /// </remarks>
    [FactorMethod(typeof(EnumSupport), "AreEqual<>")]
    [Quality(QualityBand.Stable)]
    public class DiscreteEnumAreEqualOp<TEnum>
    {
        private static int ToInt(TEnum en)
        {
            return (int)(object)en;
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(areEqual,b) p(areEqual,b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static DiscreteEnum<TEnum> AAverageConditional([SkipIfUniform] Bernoulli areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageConditional(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(b) p(b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        public static DiscreteEnum<TEnum> AAverageConditional(bool areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageConditional(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        public static DiscreteEnum<TEnum> AAverageLogarithm(bool areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageLogarithm(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(b) p(b) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static DiscreteEnum<TEnum> AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageLogarithm(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(areEqual,a) p(areEqual,a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static DiscreteEnum<TEnum> BAverageConditional([SkipIfUniform] Bernoulli areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageConditional(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a) p(a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        public static DiscreteEnum<TEnum> BAverageConditional(bool areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageConditional(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static DiscreteEnum<TEnum> BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageLogarithm(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        public static DiscreteEnum<TEnum> BAverageLogarithm(bool areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageLogarithm(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(a,b) p(a,b) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageConditional(TEnum A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.AreEqualAverageConditional(ToInt(A), B.GetInternalDiscrete());
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(a,b) p(a,b) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageConditional(DiscreteEnum<TEnum> A, TEnum B)
        {
            return DiscreteAreEqualOp.AreEqualAverageConditional(A.GetInternalDiscrete(), ToInt(B));
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(areEqual,a,b)]</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageLogarithm(TEnum A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.AreEqualAverageLogarithm(ToInt(A), B.GetInternalDiscrete());
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(areEqual,a,b)]</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageLogarithm(DiscreteEnum<TEnum> A, TEnum B)
        {
            return DiscreteAreEqualOp.AreEqualAverageLogarithm(A.GetInternalDiscrete(), ToInt(B));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual,a,b) p(areEqual,a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli areEqual, TEnum A, TEnum B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, ToInt(A), ToInt(B));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, TEnum A, TEnum B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, ToInt(A), ToInt(B));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, TEnum A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, ToInt(A), B.GetInternalDiscrete());
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, DiscreteEnum<TEnum> A, TEnum B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, A.GetInternalDiscrete(), ToInt(B));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Outgoing message to <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, TEnum A, DiscreteEnum<TEnum> B, [Fresh] DiscreteEnum<TEnum> to_B)
        {
            return DiscreteAreEqualOp.LogEvidenceRatio(areEqual, ToInt(A), B.GetInternalDiscrete(), to_B.GetInternalDiscrete());
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, DiscreteEnum<TEnum> A, TEnum B, [Fresh] DiscreteEnum<TEnum> to_A)
        {
            return DiscreteAreEqualOp.LogEvidenceRatio(areEqual, A.GetInternalDiscrete(), ToInt(B), to_A.GetInternalDiscrete());
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, TEnum A, TEnum B)
        {
            return DiscreteAreEqualOp.LogEvidenceRatio(areEqual, ToInt(A), ToInt(B));
        }
    }
}
