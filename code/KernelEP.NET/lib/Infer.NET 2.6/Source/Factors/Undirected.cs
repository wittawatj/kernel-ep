// (C) Copyright 2009-2010 Microsoft Research Cambridge

// Reference: Dan Huttenlocher - Distance Transform paper

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;

    /// <summary>
    /// Provides useful factors for undirected models.
    /// </summary>
    public class Undirected
    {
        /// <summary>
        /// Implements an integer Potts potential which has the value of 1 if a=b and exp(-logCost) otherwise.
        /// </summary>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Potts_model
        /// </remarks>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logCost"></param>
        public static void Potts(int a, int b, double logCost)
        {
            throw new NotImplementedException("Deterministic form of Potts not yet implemented");
        }

        /// <summary>
        /// Implements an boolean Potts potential which has the value of 1 if a=b and exp(-logCost) otherwise.
        /// </summary>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Potts_model
        /// </remarks>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logCost"></param>
        public static void Potts(bool a, bool b, double logCost)
        {
            throw new NotImplementedException("Deterministic form of Potts not yet implemented");
        }


        /// <summary>
        /// Implements a linear difference potential which has the value of exp( -|a-b|* logUnitCost ).
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logUnitCost"></param>
        public static void Linear(int a, int b, double logUnitCost)
        {
            throw new NotImplementedException("Deterministic form of linear difference not yet implemented");
        }

        /// <summary>
        /// Implements a truncated linear difference potential which has the value of  exp( - min( |a-b|* logUnitCost, maxCost) ).
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logUnitCost"></param>
        /// <param name="maxCost"></param>
        public static void LinearTrunc(int a, int b, double logUnitCost, double maxCost)
        {
            throw new NotImplementedException("Deterministic form of truncated linear difference not yet implemented");
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Undirected.Potts(int, int, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Undirected), "Potts", typeof(int), typeof(int), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public class PottsIntOp
    {
        //-- Max product ----------------------------------------------------------------------

        /// <summary />
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="logCost">Constant value for <c>logCost</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static UnnormalizedDiscrete AMaxConditional(UnnormalizedDiscrete B, double logCost, UnnormalizedDiscrete result)
        {
            double max = B.GetWorkspace().Max();
            double[] source = B.GetWorkspace().SourceArray;
            double[] target = result.GetWorkspace().SourceArray;
            for (int i = 0; i < target.Length; i++)
                target[i] = Math.Max(max - logCost, source[i]);
            return result;
        }

        /// <summary />
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="logCost">Constant value for <c>logCost</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static UnnormalizedDiscrete BMaxConditional(UnnormalizedDiscrete A, double logCost, UnnormalizedDiscrete result)
        {
            return AMaxConditional(A, logCost, result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Undirected.Potts(bool, bool, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Undirected), "Potts", typeof(bool), typeof(bool), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public class PottsBoolOp
    {
        //-- Max product ----------------------------------------------------------------------

        /// <summary />
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="logCost">Constant value for <c>logCost</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Bernoulli AMaxConditional(Bernoulli B, double logCost)
        {
            return Bernoulli.FromLogOdds(Math.Min(B.LogOdds, logCost));
        }

        /// <summary />
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="logCost">Constant value for <c>logCost</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Bernoulli BMaxConditional(Bernoulli A, double logCost)
        {
            return AMaxConditional(A, logCost);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Undirected.Linear(int, int, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Undirected), "Linear")]
    [Quality(QualityBand.Experimental)]
    public class LinearOp
    {
        //-- Max product ----------------------------------------------------------------------
        
        /// <summary />
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="logUnitCost">Constant value for <c>logUnitCost</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static UnnormalizedDiscrete AMaxConditional(UnnormalizedDiscrete B, double logUnitCost, UnnormalizedDiscrete result)
        {
            double[] source = B.GetWorkspace().SourceArray;
            double[] target = result.GetWorkspace().SourceArray;
            // forward pass
            target[0] = source[0];
            for (int i = 1; i < target.Length; i++)
            {
                target[i] = Math.Max(source[i], target[i - 1] - logUnitCost);
            }
            // reverse pass
            for (int i = target.Length - 2; i >= 0; i--)
            {
                target[i] = Math.Max(target[i], target[i + 1] - logUnitCost);
            }
            return result;
        }

        /// <summary />
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="logUnitCost">Constant value for <c>logUnitCost</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static UnnormalizedDiscrete BMaxConditional(UnnormalizedDiscrete A, double logUnitCost, UnnormalizedDiscrete result)
        {
            return AMaxConditional(A, logUnitCost, result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Undirected.LinearTrunc(int, int, double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Undirected), "LinearTrunc")]
    [Quality(QualityBand.Experimental)]
    public class LinearTruncOp
    {
        //-- Max product ----------------------------------------------------------------------
        
        /// <summary />
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="logUnitCost">Constant value for <c>logUnitCost</c>.</param>
        /// <param name="maxCost">Constant value for <c>maxCost</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static UnnormalizedDiscrete AMaxConditional(UnnormalizedDiscrete B, double logUnitCost, double maxCost, UnnormalizedDiscrete result)
        {
            double[] source = B.GetWorkspace().SourceArray;
            double[] target = result.GetWorkspace().SourceArray;
            // forward pass
            target[0] = source[0];
            double max = source[0];
            for (int i = 1; i < target.Length; i++)
            {
                max = Math.Max(max, source[i]);
                target[i] = Math.Max(source[i], target[i - 1] - logUnitCost);
            }
            // reverse pass
            double maxLessCost = max - maxCost;
            target[target.Length - 1] = Math.Max(target[target.Length - 1], maxLessCost);
            for (int i = target.Length - 2; i >= 0; i--)
            {
                target[i] = Math.Max(Math.Max(target[i], target[i + 1] - logUnitCost), maxLessCost);
            }
            return result;
        }

        /// <summary />
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="logUnitCost">Constant value for <c>logUnitCost</c>.</param>
        /// <param name="maxCost">Constant value for <c>maxCost</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static UnnormalizedDiscrete BMaxConditional(UnnormalizedDiscrete A, double logUnitCost, double maxCost, UnnormalizedDiscrete result)
        {
            return AMaxConditional(A, logUnitCost, maxCost, result);
        }
    }
}
