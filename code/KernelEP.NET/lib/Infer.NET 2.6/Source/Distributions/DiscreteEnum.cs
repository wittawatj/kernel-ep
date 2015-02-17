// (C) Copyright 2009-2010 Microsoft Research Cambridge

using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Factors;

namespace MicrosoftResearch.Infer.Distributions
{
    /// <summary>
    /// A discrete distribution over the values of an enum.
    /// </summary>
    /// <typeparam name="TEnum"></typeparam>
    [Quality(QualityBand.Preview)]
    public class DiscreteEnum<TEnum> : GenericDiscreteBase<TEnum, DiscreteEnum<TEnum>>
    {
        private static Array values = Enum.GetValues(typeof (TEnum));

        /// <summary>
        /// Creates a uniform distribution over the enum values.
        /// </summary>
        public DiscreteEnum() :
            base(values.Length, Sparsity.Dense)
        {
        }

        /// <summary>
        /// Creates a distribution over the enum values using the specified probs.
        /// </summary>
        public DiscreteEnum(params double[] probs) :
            base(values.Length, Sparsity.Dense)
        {
            disc.SetProbs(Vector.FromArray(probs));
        }

        /// <summary>
        /// Converts from an integer to an enum value
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        protected override TEnum ConvertFromInt(int i)
        {
            return (TEnum) values.GetValue(i);
        }

        /// <summary>
        /// Converts the enum value to an integer
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        protected override int ConvertToInt(TEnum value)
        {
            return (int) (object) value;
        }
    }
}
