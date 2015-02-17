// (C) Copyright 2008 Microsoft Research Cambridge

#define SpecializeArrays
#define MinimalGenericTypeParameters

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.UsesEqualDef{T}(T, int, out T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefOp
    {
        /// <summary>
        /// Evidence message for EP
        /// </summary>
        /// <param name="Uses">Incoming message from 'Uses'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="Def">Incoming message from 'Def'.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence</returns>
        /// <remarks><para>
        /// The formula for the result is <c>log(sum_(Uses,Def) p(Uses,Def) factor(Uses,Def,Marginal) / sum_Uses p(Uses) messageTo(Uses))</c>.
        /// Adding up these values across all factors and variables gives the log-evidence estimate for EP.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Uses"/> is not a proper distribution</exception>
        public static double LogEvidenceRatio1<T>([SkipIfAllUniform] IList<T> Uses, T Def)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, SettableTo<T>, ICloneable, SettableToUniform
        {
            if (Uses.Count <= 1)
                return 0.0;
            else
            {
                T toUse = (T)Def.Clone();
                T[] productBefore = new T[Uses.Count];
                T productAfter = (T)Def.Clone();
                productAfter.SetToUniform();
                double z = 0.0;
                for (int i = 0; i < Uses.Count; i++)
                {
                    productBefore[i] = (T)Def.Clone();
                    if (i > 0)
                        productBefore[i].SetToProduct(productBefore[i - 1], Uses[i - 1]);
                    z += productBefore[i].GetLogAverageOf(Uses[i]);
                }
                // z is now log(sum_x Def(x)*prod_i Uses[i](x))
                for (int i = Uses.Count - 1; i >= 0; i--)
                {
                    toUse.SetToProduct(productBefore[i], productAfter);
                    z -= toUse.GetLogAverageOf(Uses[i]);
                    productAfter.SetToProduct(productAfter, Uses[i]);
                }
                return z;
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="to_Uses">Outgoing message to <c>Uses</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Uses,Def) p(Uses,Def) factor(Uses,Def,count,Marginal) / sum_Uses p(Uses) messageTo(Uses))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static double LogEvidenceRatio<T>([SkipIfAllUniform] IList<T> Uses, T Def, [Fresh] IList<T> to_Uses)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, SettableTo<T>, ICloneable, SettableToUniform
        {
            if (Uses.Count <= 1)
                return 0.0;
            else
            {
                T productBefore = (T)Def.Clone();
                double z = 0.0;
                T previous_use = Def;
                for (int i = 0; i < Uses.Count; i++)
                {
                    if (i > 0)
                        productBefore.SetToProduct(productBefore, previous_use);
                    T use = Uses[i];
                    z += productBefore.GetLogAverageOf(use);
                    // z is now log(sum_x Def(x)*prod_i Uses[i](x))
                    z -= to_Uses[i].GetLogAverageOf(use);
                    previous_use = use;
                }
                return z;
            }
        }

        /// <summary>EP message to <c>Marginal</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Marginal</c> as the random arguments are varied. The formula is <c>proj[p(Marginal) sum_(Uses,Def) p(Uses,Def) factor(Uses,Def,count,Marginal)]/p(Marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] IList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

#if SpecializeArrays
        /// <summary>EP message to <c>Marginal</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Marginal</c> as the random arguments are varied. The formula is <c>proj[p(Marginal) sum_(Uses,Def) p(Uses,Def) factor(Uses,Def,count,Marginal)]/p(Marginal)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] T[] Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }
#endif

        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,count,Marginal)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // TM: SkipIfUniform on Def added as a stronger constraint, to prevent improper messages in EP.
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] IList<T> Uses, [SkipIfAllUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException("resultIndex");
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }

#if SpecializeArrays
        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,count,Marginal)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // TM: SkipIfUniform on Def added as a stronger constraint, to prevent improper messages in EP.
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] T[] Uses, [SkipIfAllUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException("resultIndex");
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }
#endif

#if MinimalGenericTypeParameters
        /// <summary>EP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Def</c> as the random arguments are varied. The formula is <c>proj[p(Def) sum_(Uses) p(Uses) factor(Uses,Def,count,Marginal)]/p(Def)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] IList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#else
			public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] IList<TUses> Uses, T result)
						where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
				{
						return Distribution.SetToProductOfAll(result, Uses);
				}
#endif
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.UsesEqualDef{T}(T, int, out T)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable.</typeparam>
    [FactorMethod(typeof(Factor), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefGibbsOp<T>
    {
        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, TDist Def, GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, Sampleable<T>, CanGetLogAverageOf<TDist>, SettableTo<TDist>, SettableToProduct<TDist>
        {
            if (Uses.Count == 1)
            {
                // the total evidence contribution of this variable should be Def.GetLogAverageOf(Uses[0]).
                // but since this variable is sending a sample to Def and Use, and those factors will send their own evidence contribution,
                // we need to cancel the contribution of those factors here.
                return Def.GetLogAverageOf(Uses[0]) - Def.GetLogProb(to_marginal.LastSample) - Uses[0].GetLogProb(to_marginal.LastSample);
            }
            else
            {
                //throw new ApplicationException("Gibbs Sampling does not support variables defined within a gate");
                double z = 0.0;
                TDist productBefore = (TDist)Def.Clone();
                TDist product = (TDist)Def.Clone();
                for (int i = 0; i < Uses.Count; i++)
                {
                    if (i > 0)
                        product.SetToProduct(productBefore, Uses[i - 1]);
                    z += product.GetLogAverageOf(Uses[i]);
                    productBefore.SetTo(product);
                }
                // z is now log(sum_x Def(x)*prod_i Uses[i](x)), which is the desired total evidence.
                // but we must also cancel the contribution of the parent and child factors that received a sample from us.
                z -= Def.GetLogProb(to_marginal.LastSample);
                for (int i = 0; i < Uses.Count; i++)
                {
                    z -= Uses[i].GetLogProb(to_marginal.LastSample);
                }
                return z;
            }
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Stochastic]
        //[SkipIfAllUniform("Uses","Def")]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IList<TDist> Uses,
            [Proper] TDist Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.SetTo(Def);
            marginal = Distribution.SetToProductWithAll(marginal, Uses);
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Stochastic]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IList<TDist> Uses,
            T Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Def;
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Stochastic]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IList<T> Uses,
            TDist Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            if (Uses.Count > 1)
                throw new ArgumentException("Uses.Count > 1");
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Uses[0];
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Skip]
        public static GibbsMarginal<TDist, T> MarginalGibbsInit<TDist>([IgnoreDependency] TDist Def)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return new GibbsMarginal<TDist, T>(Def, 100, 1, true, true, true);
        }

        /// <summary />
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, int resultIndex, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return to_marginal.LastSample;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static TDist UsesGibbs<TDist>(
            [IgnoreDependency] ICollection<TDist> Uses,
            [IsReturned] TDist Def,
            int resultIndex, TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException("resultIndex");
            if (Uses.Count > 1)
                throw new ArgumentException("Uses.Count > 1");
            result.SetTo(Def);
            return result;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="count">Constant value for <c>count</c>.</param>
        /// <param name="factory" />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TArrayType">The type of arrays produced by <paramref name="factory"/>.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        [Skip]
        public static TArrayType UsesGibbsInit<TArrayType, TDef>(
            [IgnoreDependency] TDef Def, int count, IArrayFactory<TDef, TArrayType> factory)
            where TDef : ICloneable
        {
            return factory.CreateArray(count, i => (TDef)Def.Clone());
        }

        /// <summary />
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static T DefGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return to_marginal.LastSample;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        //[MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] IList<TDist> Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            result.SetToUniform();
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.UsesEqualDefGibbs{T}(T, int, int, int, out T, out T, out T)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable.</typeparam>
    [FactorMethod(typeof(Factor), "UsesEqualDefGibbs<>")]
    [Buffers("sample", "conditional", "marginalEstimator", "sampleAcc", "conditionalAcc")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefGibbsOp2<T>
    {
        /// <summary>Initialize the buffer <c>conditional</c>.</summary>
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <returns>Initial value of buffer <c>conditional</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static TDist ConditionalInit<TDist>([IgnoreDependency] TDist def)
            where TDist : ICloneable
        {
            return (TDist)def.Clone();
        }

        /// <summary>Update the buffer <c>conditional</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static TDist Conditional<TDist>(IList<TDist> Uses, [SkipIfAnyUniform] TDist Def, TDist result)
            where TDist : SettableTo<TDist>, SettableToProduct<TDist>
        {
            result.SetTo(Def);
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }

        /// <summary>Update the buffer <c>sample</c>.</summary>
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="conditional">Buffer <c>conditional</c>.</param>
        /// <returns>New value of buffer <c>sample</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Stochastic]
        public static T Sample<TDist>([IgnoreDependency] TDist def, [Proper] TDist conditional)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return conditional.Sample();
        }

        /// <summary>Initialize the buffer <c>marginalEstimator</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="burnIn">Constant value for <c>burnIn</c>.</param>
        /// <returns>Initial value of buffer <c>marginalEstimator</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static BurnInAccumulator<TDist> MarginalEstimatorInit<TDist>([IgnoreDependency] TDist to_marginal, int burnIn)
            where TDist : IDistribution<T>
        {
            Accumulator<TDist> est = (Accumulator<TDist>)ArrayEstimator.CreateEstimator<TDist, T>(to_marginal, true);
            return new BurnInAccumulator<TDist>(burnIn, 1, est);
        }

        /// <summary>Update the buffer <c>marginalEstimator</c>.</summary>
        /// <param name="conditional">Buffer <c>conditional</c>.</param>
        /// <param name="marginalEstimator">Buffer <c>marginalEstimator</c>.</param>
        /// <returns>New value of buffer <c>marginalEstimator</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static TAcc MarginalEstimator<TDist, TAcc>([Proper] TDist conditional, TAcc marginalEstimator)
            where TAcc : Accumulator<TDist>
        {
            marginalEstimator.Add(conditional);
            return marginalEstimator;
        }

        /// <summary />
        /// <param name="marginalEstimator">Buffer <c>marginalEstimator</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static TDist MarginalGibbs<TDist>(BurnInAccumulator<TDist> marginalEstimator, TDist result)
        {
            return ((Estimator<TDist>)marginalEstimator.Accumulator).GetDistribution(result);
        }

        /// <summary>Initialize the buffer <c>sampleAcc</c>.</summary>
        /// <param name="to_samples">Previous outgoing message to <c>samples</c>.</param>
        /// <param name="burnIn">Constant value for <c>burnIn</c>.</param>
        /// <param name="thin">Constant value for <c>thin</c>.</param>
        /// <returns>Initial value of buffer <c>sampleAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Accumulator<T> SampleAccInit(ICollection<T> to_samples, int burnIn, int thin)
        {
            return new BurnInAccumulator<T>(burnIn, thin, new AccumulateIntoCollection<T>(to_samples));
        }

        /// <summary>Update the buffer <c>sampleAcc</c>.</summary>
        /// <param name="sample">Buffer <c>sample</c>.</param>
        /// <param name="sampleAcc">Buffer <c>sampleAcc</c>.</param>
        /// <returns>New value of buffer <c>sampleAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Accumulator<T> SampleAcc(T sample, Accumulator<T> sampleAcc)
        {
            sampleAcc.Add(sample);
            return sampleAcc;
        }

        /// <summary />
        /// <param name="sampleAcc">Buffer <c>sampleAcc</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static TList SamplesGibbs<TList>(Accumulator<T> sampleAcc, TList result)
            where TList : ICollection<T>
        {
            // do nothing since result was already modified by sampleAcc
            return result;
        }

        /// <summary>Initialize the buffer <c>conditionalAcc</c>.</summary>
        /// <param name="to_conditionals">Previous outgoing message to <c>conditionals</c>.</param>
        /// <param name="burnIn">Constant value for <c>burnIn</c>.</param>
        /// <param name="thin">Constant value for <c>thin</c>.</param>
        /// <returns>Initial value of buffer <c>conditionalAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Accumulator<TDist> ConditionalAccInit<TDist>(ICollection<TDist> to_conditionals, int burnIn, int thin)
        {
            return new BurnInAccumulator<TDist>(burnIn, thin, new AccumulateIntoCollection<TDist>(to_conditionals));
        }

        /// <summary>Update the buffer <c>conditionalAcc</c>.</summary>
        /// <param name="conditional">Buffer <c>conditional</c>.</param>
        /// <param name="conditionalAcc">Buffer <c>conditionalAcc</c>.</param>
        /// <returns>New value of buffer <c>conditionalAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Accumulator<TDist> ConditionalAcc<TDist>(TDist conditional, Accumulator<TDist> conditionalAcc)
            where TDist : ICloneable
        {
            conditionalAcc.Add((TDist)conditional.Clone());
            return conditionalAcc;
        }

        /// <summary />
        /// <param name="conditionalAcc">Buffer <c>conditionalAcc</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static TDistList ConditionalsGibbs<TDist, TDistList>(Accumulator<TDist> conditionalAcc, TDistList result)
            where TDistList : ICollection<TDist>
        {
            // do nothing since result was already modified by Acc
            return result;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="sample">Buffer <c>sample</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, TDist Def, T sample)
            where TDist : IDistribution<T>, Sampleable<T>, CanGetLogAverageOf<TDist>, SettableTo<TDist>, SettableToProduct<TDist>
        {
            if (Uses.Count == 1)
            {
                // the total evidence contribution of this variable should be Def.GetLogAverageOf(Uses[0]).
                // but since this variable is sending a sample to Def and Use, and those factors will send their own evidence contribution,
                // we need to cancel the contribution of those factors here.
                return Def.GetLogAverageOf(Uses[0]) - Def.GetLogProb(sample) - Uses[0].GetLogProb(sample);
            }
            else
            {
                //throw new ApplicationException("Gibbs Sampling does not support variables defined within a gate");
                double z = 0.0;
                TDist productBefore = (TDist)Def.Clone();
                TDist product = (TDist)Def.Clone();
                for (int i = 0; i < Uses.Count; i++)
                {
                    if (i > 0)
                        product.SetToProduct(productBefore, Uses[i - 1]);
                    z += product.GetLogAverageOf(Uses[i]);
                    productBefore.SetTo(product);
                }
                // z is now log(sum_x Def(x)*prod_i Uses[i](x)), which is the desired total evidence.
                // but we must also cancel the contribution of the parent and child factors that received a sample from us.
                z -= Def.GetLogProb(sample);
                for (int i = 0; i < Uses.Count; i++)
                {
                    z -= Uses[i].GetLogProb(sample);
                }
                return z;
            }
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="sample">Buffer <c>sample</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static T UsesGibbs<TDist>(TDist def, T sample, int resultIndex, T result)
            where TDist : IDistribution<T>
        {
            return sample;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static TDist UsesGibbs<TDist>(
            [IgnoreDependency] ICollection<TDist> Uses,
            [IsReturned] TDist Def,
            int resultIndex, TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException("resultIndex");
            if (Uses.Count > 1)
                throw new ArgumentException("Uses.Count > 1");
            result.SetTo(Def);
            return result;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Skip]
        public static TDist UsesGibbsInit<TDist>([IgnoreDependency] TDist Def, int resultIndex)
            where TDist : ICloneable
        {
            return (TDist)Def.Clone();
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="sample">Buffer <c>sample</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static T DefGibbs<TDist>(TDist def, [IsReturned] T sample)
            where TDist : IDistribution<T>
        {
            return sample;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        //[MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] IList<TDist> Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            result.SetToUniform();
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.UsesEqualDef{T}(T, int, out T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefMaxOp
    {
        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T UsesMaxConditional<T>([AllExceptIndex] IList<T> Uses, [SkipIfUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = UsesEqualDefOp.UsesAverageConditional<T>(Uses, Def, resultIndex, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefMaxConditional<T>([SkipIfAllUniform] IList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return UsesEqualDefOp.DefAverageConditional<T>(Uses, result);
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T MarginalMaxConditional<T>(IList<T> Uses, [SkipIfUniform] T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = UsesEqualDefOp.MarginalAverageConditional<T>(Uses, Def, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.UsesEqualDef{T}(T, int, out T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "UsesEqualDef<>", Default = true)]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefVmpBufferOp
    {
        /// <summary>VMP message to <c>Marginal</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Marginal</c>. The formula is <c>exp(sum_(Uses,Def) p(Uses,Def) log(factor(Uses,Def,count,Marginal)))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageLogarithm<T>([NoInit] IList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

        /// <summary>VMP message to <c>Uses</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Uses</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T UsesAverageLogarithm<T>([IsReturned] T to_marginal, int resultIndex, T result)
            where T : SettableTo<T>
        {
            result.SetTo(to_marginal);
            return result;
        }

        /// <summary>VMP message to <c>Def</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Def</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageLogarithm<T>([IsReturned] T to_marginal, T result)
            where T : SettableTo<T>
        {
            result.SetTo(to_marginal);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.UsesEqualDef{T}(T, int, out T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefVmpOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <param name="to_marginal">Outgoing message to <c>marginal</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Uses,Def,count,Marginal))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static double AverageLogFactor<T>([Fresh] T to_marginal)
            where T : CanGetAverageLog<T>
        {
            return -to_marginal.GetAverageLog(to_marginal);
        }

#if MinimalGenericTypeParameters
        /// <summary>VMP message to <c>Marginal</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Marginal</c>. The formula is <c>exp(sum_(Uses,Def) p(Uses,Def) log(factor(Uses,Def,count,Marginal)))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T MarginalAverageLogarithm<T>(IList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            return UsesAverageLogarithm(Uses, Def, 0, result);
        }

        /// <summary>VMP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Uses</c>. The formula is <c>exp(sum_(Def) p(Def) log(factor(Uses,Def,count,Marginal)))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T UsesAverageLogarithm<T>([NoInit] IList<T> Uses, T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

        /// <summary>VMP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Def</c>. The formula is <c>exp(sum_(Uses) p(Uses) log(factor(Uses,Def,count,Marginal)))</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // TM: Proper added on Def to avoid improper messages.
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T DefAverageLogarithm<T>([NoInit] IList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            return UsesAverageLogarithm(Uses, Def, 0, result);
        }
#else
		[SkipIfAllUniform]
		public static T MarginalAverageLogarithm<T, TUses, TDef>(IList<TUses> Uses, TDef Def, T result)
				where T : SettableToProduct<TUses>, SettableTo<TDef>, TUses
		{
			return UsesAverageLogarithm<T, TUses, TDef>(Uses, Def, 0, result);
		}

		[SkipIfAllUniform]
		public static T UsesAverageLogarithm<T, TUses, TDef>([MatchingIndexTrigger] IList<TUses> Uses, TDef Def, int resultIndex, T result)
					where T : SettableToProduct<TUses>, SettableTo<TDef>, TUses
		{
			result.SetTo(Def);
			return Distribution.SetToProductWithAll(result, Uses);
		}

		[SkipIfAllUniform]
		public static T DefAverageLogarithm<T, TUses, TDef>(IList<TUses> Uses, [Trigger] TDef Def, T result)
					where T : SettableToProduct<TUses>, SettableTo<TDef>, TUses
		{
			return UsesAverageLogarithm<T, TUses, TDef>(Uses, Def, 0, result);
		}
#endif
    }
}
