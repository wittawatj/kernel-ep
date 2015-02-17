using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MicrosoftResearch.Infer.Factors
{
    // TODO: make a factor for this
    [Quality(QualityBand.Preview)]
    internal static class IsGreaterThanDoubleOp
    {
        public static Bernoulli IsGreaterThanAverageConditional([Proper] Beta a, double b)
        {
            if (a.IsPointMass)
                return Bernoulli.PointMass(a.Point > b);
            return new Bernoulli(1 - a.GetProbLessThan(b));
        }

        public static Bernoulli IsGreaterThanAverageConditional(double a, [Proper] Beta b)
        {
            if (b.IsPointMass)
                return Bernoulli.PointMass(a > b.Point);
            return new Bernoulli(b.GetProbLessThan(a));
        }
    }
}
