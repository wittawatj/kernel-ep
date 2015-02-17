// (C) Copyright 2009 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System.Collections.Generic;
    using System.Linq;

    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;

    /// <summary>
    /// Holds messages for IndexOfMaximumOp
    /// </summary>
    public class IndexOfMaximumBuffer
    {
        public IList<Gaussian> MessagesToMax;
        public IList<Gaussian> to_list;
    }


    /// <summary>Provides outgoing messages for <see cref="MMath.IndexOfMaximumDouble(IList{double})" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(MMath), "IndexOfMaximumDouble")]
    [Quality(QualityBand.Experimental)]
    [Buffers("Buffer")]
    public static class IndexOfMaximumOp
    {
        /// <summary>Initialize the buffer <c>Buffer</c>.</summary>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <returns>Initial value of buffer <c>Buffer</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static IndexOfMaximumBuffer BufferInit<GaussianList>([IgnoreDependency] GaussianList list)
            where GaussianList : IList<Gaussian>
        {
            return new IndexOfMaximumBuffer
                {
                    MessagesToMax = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray()),
                    to_list = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray())
                };
        }

        /// <summary>Update the buffer <c>Buffer</c>.</summary>
        /// <param name="Buffer">Buffer <c>Buffer</c>.</param>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <param name="IndexOfMaximumDouble">Constant value for <c>indexOfMaximumDouble</c>.</param>
        /// <returns>New value of buffer <c>Buffer</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static IndexOfMaximumBuffer Buffer<GaussianList>(
            IndexOfMaximumBuffer Buffer, GaussianList list, int IndexOfMaximumDouble) // redundant parameters required for correct dependency graph
            where GaussianList : IList<Gaussian>
        {
            var max_marginal = Buffer.to_list[IndexOfMaximumDouble] * list[IndexOfMaximumDouble];
            Gaussian product = Gaussian.Uniform();
            //var order = Rand.Perm(list.Count); 
            for (int i = 0; i < list.Count; i++)
            {
                //int c = order[i]; 
                int c = i;
                if (c != IndexOfMaximumDouble)
                {
                    var msg_to_sum = max_marginal / Buffer.MessagesToMax[c];

                    var msg_to_positiveop = DoublePlusOp.AAverageConditional(Sum: msg_to_sum, b: list[c]);
                    var msgFromPositiveOp = IsPositiveOp.XAverageConditional(true, msg_to_positiveop);
                    Buffer.MessagesToMax[c] = DoublePlusOp.SumAverageConditional(list[c], msgFromPositiveOp);
                    Buffer.to_list[c] = DoublePlusOp.AAverageConditional(Sum: msg_to_sum, b: msgFromPositiveOp);
                    max_marginal = msg_to_sum * Buffer.MessagesToMax[c];
                    product.SetToProduct(product, Buffer.MessagesToMax[c]);
                }
            }
            //Buffer.to_list[IndexOfMaximumDouble] = max_marginal / list[IndexOfMaximumDouble];
            Buffer.to_list[IndexOfMaximumDouble] = product;
            return Buffer;
        }

        /// <summary>EP message to <c>list</c>.</summary>
        /// <param name="Buffer">Buffer <c>Buffer</c>.</param>
        /// <param name="to_list">Previous outgoing message to <c>list</c>.</param>
        /// <param name="IndexOfMaximumDouble">Constant value for <c>indexOfMaximumDouble</c>.</param>
        /// <returns>The outgoing EP message to the <c>list</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>list</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static GaussianList listAverageConditional<GaussianList>(
            IndexOfMaximumBuffer Buffer, GaussianList to_list, int IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            to_list.SetTo(Buffer.to_list);
            return to_list;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Buffer">Buffer <c>Buffer</c>.</param>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <param name="IndexOfMaximumDouble">Constant value for <c>indexOfMaximumDouble</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(list) p(list) factor(indexOfMaximumDouble,list))</c>.</para>
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogAverageFactor<GaussianList>(IndexOfMaximumBuffer Buffer, GaussianList list, int IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            double evidence = 0;
            var max_marginal = list[IndexOfMaximumDouble] * Buffer.to_list[IndexOfMaximumDouble];
            for (int c = 0; c < list.Count; c++)
            {
                if (c != IndexOfMaximumDouble)
                {
                    var msg_to_sum = max_marginal / Buffer.MessagesToMax[c];
                    var msg_to_positiveop = DoublePlusOp.AAverageConditional(Sum: msg_to_sum, b: list[c]);
                    evidence += IsPositiveOp.LogEvidenceRatio(true, msg_to_positiveop);
                    // sum operator does not contribute because no projection is involved
                    // the x[index]-x[c] variable does not contribute because it only connects to two factors
                    evidence -= msg_to_sum.GetLogAverageOf(Buffer.MessagesToMax[c]);
                    if (max_marginal.IsPointMass)
                        evidence += Buffer.MessagesToMax[c].GetLogAverageOf(max_marginal);
                    else
                        evidence -= Buffer.MessagesToMax[c].GetLogNormalizer();
                }
            }
            //evidence += ReplicateOp.LogEvidenceRatio<Gaussian>(MessagesToMax, list[IndexOfMaximumDouble], MessagesToMax.Select(o => max_marginal / o).ToArray());
            if (!max_marginal.IsPointMass)
                evidence += max_marginal.GetLogNormalizer() - list[IndexOfMaximumDouble].GetLogNormalizer();
            //evidence -= Buffer.MessagesToMax.Sum(o => o.GetLogNormalizer());
            //evidence -= Buffer.MessagesToMax.Sum(o => (max_marginal / o).GetLogAverageOf(o));
            return evidence;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Buffer">Buffer <c>Buffer</c>.</param>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <param name="IndexOfMaximumDouble">Constant value for <c>indexOfMaximumDouble</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(list) p(list) factor(indexOfMaximumDouble,list))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogEvidenceRatio<GaussianList>(IndexOfMaximumBuffer Buffer, GaussianList list, int IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            return LogAverageFactor(Buffer, list, IndexOfMaximumDouble);
        }
    }


    /// <summary>Provides outgoing messages for <see cref="MMath.IndexOfMaximumDouble(IList{double})" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(MMath), "IndexOfMaximumDouble", Default = true)]
    [Quality(QualityBand.Experimental)]
    [Buffers("Buffers")]
    public static class IndexOfMaximumStochasticOp
    {
        /// <summary>Initialize the buffer <c>Buffers</c>.</summary>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <param name="IndexOfMaximumDouble">Incoming message from <c>indexOfMaximumDouble</c>.</param>
        /// <returns>Initial value of buffer <c>Buffers</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static IndexOfMaximumBuffer[] BuffersInit<GaussianList>([IgnoreDependency] GaussianList list, [IgnoreDependency] Discrete IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            return list.Select(o => IndexOfMaximumOp.BufferInit(list)).ToArray();
        }

        /// <summary>Update the buffer <c>Buffers</c>.</summary>
        /// <param name="Buffers">Buffer <c>Buffers</c>.</param>
        /// <param name="list">Incoming message from <c>list</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="IndexOfMaximumDouble">Incoming message from <c>indexOfMaximumDouble</c>.</param>
        /// <returns>New value of buffer <c>Buffers</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="list" /> is not a proper distribution.</exception>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static IndexOfMaximumBuffer[] Buffers<GaussianList>(IndexOfMaximumBuffer[] Buffers, [SkipIfUniform] GaussianList list, Discrete IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            for (int i = 0; i < list.Count; i++)
            {
                Buffers[i].to_list = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray());
                Buffers[i].to_list[i] = Buffers[i].MessagesToMax.Aggregate((p, q) => p * q);
                Buffers[i] = IndexOfMaximumOp.Buffer(Buffers[i], list, i);
            }
            return Buffers;
        }

        /// <summary>EP message to <c>list</c>.</summary>
        /// <param name="Buffers">Buffer <c>Buffers</c>.</param>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <param name="IndexOfMaximumDouble">Incoming message from <c>indexOfMaximumDouble</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>list</c> as the random arguments are varied. The formula is <c>proj[p(list) sum_(indexOfMaximumDouble) p(indexOfMaximumDouble) factor(indexOfMaximumDouble,list)]/p(list)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="IndexOfMaximumDouble" /> is not a proper distribution.</exception>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static GaussianList listAverageConditional<GaussianList>(
            [Fresh, SkipIfUniform] IndexOfMaximumBuffer[] Buffers, GaussianList list, [SkipIfUniform] Discrete IndexOfMaximumDouble, GaussianList result)
            where GaussianList : DistributionStructArray<Gaussian, double> // IList<Gaussian>
        {
            // TODO: check if Index is a point mass
            var results = list.Select(o => list.Select(p => Gaussian.Uniform()).ToList()).ToArray();
            //var evidences = new Bernoulli[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                for (int j = 0; j < list.Count; j++)
                {
                    results[j][i] = Buffers[i].to_list[j];
                }
            }
            for (int i = 0; i < list.Count; i++)
            {
                result[i] = GateEnterPartialOp<double>.ValueAverageConditional<Gaussian>(results[i], IndexOfMaximumDouble, list[i],
                                                                                         System.Linq.Enumerable.Range(0, list.Count).ToArray(), result[i]);
            }
            return result;
        }

        /// <summary>EP message to <c>indexOfMaximumDouble</c>.</summary>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <param name="Buffers">Buffer <c>Buffers</c>.</param>
        /// <returns>The outgoing EP message to the <c>indexOfMaximumDouble</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>indexOfMaximumDouble</c> as the random arguments are varied. The formula is <c>proj[p(indexOfMaximumDouble) sum_(list) p(list) factor(indexOfMaximumDouble,list)]/p(indexOfMaximumDouble)</c>.</para>
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static Discrete IndexOfMaximumDoubleAverageConditional<GaussianList>(GaussianList list, IndexOfMaximumBuffer[] Buffers)
            where GaussianList : DistributionStructArray<Gaussian, double>
        {
            // var results = list.Select(o => list.Select(p => Gaussian.Uniform()).ToList()).ToArray();
            // TODO: if IndexOfMaximumDouble is uniform we will never call this routine so buffers will not get set, so messages to IndexOfMaximumDouble will be incorrect
            var evidences = new double[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                //var res = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray());
                //res[i] = Buffer[i].MessagesToMax.Aggregate((p, q) => p * q);
                evidences[i] = IndexOfMaximumOp.LogAverageFactor(Buffers[i], list, i);
            }
            return new Discrete(MMath.Softmax(evidences));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="list">Incoming message from <c>list</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_list">Previous outgoing message to <c>list</c>.</param>
        /// <param name="Buffers">Buffer <c>Buffers</c>.</param>
        /// <param name="IndexOfMaximumDouble">Incoming message from <c>indexOfMaximumDouble</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(list,indexOfMaximumDouble) p(list,indexOfMaximumDouble) factor(indexOfMaximumDouble,list))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="list" /> is not a proper distribution.</exception>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogAverageFactor<GaussianList>(
            [SkipIfUniform] GaussianList list, GaussianList to_list, IndexOfMaximumBuffer[] Buffers, Discrete IndexOfMaximumDouble)
            where GaussianList : DistributionStructArray<Gaussian, double>
        {
            var evidences = new double[list.Count];
            var tempBuffer = new IndexOfMaximumBuffer();
            for (int i = 0; i < list.Count; i++)
            {
                tempBuffer.to_list = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray());
                tempBuffer.to_list[i] = Buffers[i].MessagesToMax.Aggregate((p, q) => p * q);
                tempBuffer.MessagesToMax = new DistributionStructArray<Gaussian, double>(Buffers[i].MessagesToMax.Select(o => (Gaussian)o.Clone()).ToArray());
                evidences[i] = IndexOfMaximumOp.LogAverageFactor(tempBuffer, list, i) + IndexOfMaximumDouble.GetLogProb(i);
                ;
            }
            return MMath.LogSumExp(evidences);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="list">Incoming message from <c>list</c>.</param>
        /// <param name="to_list">Previous outgoing message to <c>list</c>.</param>
        /// <param name="IndexOfMaximumDouble">Incoming message from <c>indexOfMaximumDouble</c>.</param>
        /// <param name="to_IndexOfMaximumDouble">Previous outgoing message to <c>IndexOfMaximumDouble</c>.</param>
        /// <param name="Buffers">Buffer <c>Buffers</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(list,indexOfMaximumDouble) p(list,indexOfMaximumDouble) factor(indexOfMaximumDouble,list) / sum_indexOfMaximumDouble p(indexOfMaximumDouble) messageTo(indexOfMaximumDouble))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogEvidenceRatio<GaussianList>(
            GaussianList list, GaussianList to_list, Discrete IndexOfMaximumDouble, Discrete to_IndexOfMaximumDouble, IndexOfMaximumBuffer[] Buffers)
            where GaussianList : DistributionStructArray<Gaussian, double>
        {
            return LogAverageFactor(list, to_list, Buffers, IndexOfMaximumDouble) - IndexOfMaximumDouble.GetLogAverageOf(to_IndexOfMaximumDouble);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="MMath.IndexOfMaximumDouble(IList{double})" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(MMath), "IndexOfMaximumDouble", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class IndexOfMaximumOp_Fast
    {
        /// <summary>EP message to <c>indexOfMaximumDouble</c>.</summary>
        /// <param name="list">Incoming message from <c>list</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>indexOfMaximumDouble</c> as the random arguments are varied. The formula is <c>proj[p(indexOfMaximumDouble) sum_(list) p(list) factor(indexOfMaximumDouble,list)]/p(indexOfMaximumDouble)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="list" /> is not a proper distribution.</exception>
        public static Discrete IndexOfMaximumDoubleAverageConditional([SkipIfAnyUniform, Proper] IList<Gaussian> list, Discrete result)
        {
            // Fast approximate calculation of downward message
            
            if (list.Count <= 1)
                return result;
            Gaussian[] maxBefore = new Gaussian[list.Count];
            maxBefore[1] = list[0];
            for (int i = 2; i < list.Count; i++)
            {
                maxBefore[i] = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxBefore[i - 1], list[i - 1]);
            }
            Gaussian[] maxAfter = new Gaussian[list.Count];
            maxAfter[list.Count - 2] = list[list.Count - 1];
            for (int i = list.Count - 3; i >= 0; i--)
            {
                maxAfter[i] = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxAfter[i + 1], list[i + 1]);
            }
            Vector probs = result.GetWorkspace();
            probs[0] = ProbGreater(list[0], maxAfter[0]).GetProbTrue();
            int last = list.Count - 1;
            probs[last] = ProbGreater(list[last], maxBefore[last]).GetProbTrue();
            for (int i = 1; i < last; i++)
            {
                Gaussian maxOther = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxBefore[i], maxAfter[i]);
                probs[i] = ProbGreater(list[i], maxOther).GetProbTrue();
            }
            result.SetProbs(probs);
            return result;
        }

        /// <summary>EP message to <c>indexOfMaximumDouble</c>.</summary>
        /// <param name="list">Incoming message from <c>list</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>indexOfMaximumDouble</c> as the random arguments are varied. The formula is <c>proj[p(indexOfMaximumDouble) sum_(list) p(list) factor(indexOfMaximumDouble,list)]/p(indexOfMaximumDouble)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="list" /> is not a proper distribution.</exception>
        public static Discrete IndexOfMaximumDoubleAverageConditional2(IList<Gaussian> list, Discrete result)
        {
            // Fast approximate calculation of downward message
            
            // TODO: sort list first
            // best accuracy is achieved by processing in decreasing order of means
            Gaussian max = list[0];
            Vector probs = result.GetWorkspace();
            probs[0] = 1.0;
            for (int i = 1; i < list.Count; i++)
            {
                Gaussian A = max;
                Gaussian B = list[i];
                double pMax = ProbGreater(A, B).GetProbTrue();
                for (int j = 0; j < i; j++)
                {
                    probs[j] *= pMax;
                }
                probs[i] = 1 - pMax;
                max = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), A, B);
            }
            result.SetProbs(probs);
            return result;
        }

        /// <summary>
        /// Returns the probability that A>B
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        private static Bernoulli ProbGreater(Gaussian A, Gaussian B)
        {
            Gaussian diff = DoublePlusOp.AAverageConditional(Sum: A, b: B);
            return IsPositiveOp.IsPositiveAverageConditional(diff);
        }
    }
}
