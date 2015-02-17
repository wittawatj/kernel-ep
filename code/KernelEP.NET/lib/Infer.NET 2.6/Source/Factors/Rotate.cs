namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.Rotate(double, double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Rotate", typeof(double), typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class RotateOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <param name="rotate">Incoming message from <c>rotate</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor(VectorGaussian rotate)
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>x</c>.</summary>
        /// <param name="rotate">Incoming message from <c>rotate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="angle">Incoming message from <c>angle</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>x</c>. Because the factor is deterministic, <c>rotate</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(angle) p(angle) log(sum_rotate p(rotate) factor(rotate,x,y,angle)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rotate" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="angle" /> is not a proper distribution.</exception>
        public static Gaussian XAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] WrappedGaussian angle)
        {
            // for x ~ N(m,v):
            // E[cos(x)] = cos(m)*exp(-v/2)
            // E[sin(x)] = sin(m)*exp(-v/2)
            if (angle.Period != 2 * Math.PI)
                throw new ArgumentException("angle.Period (" + angle.Period + ") != 2*PI (" + 2 * Math.PI + ")");
            double angleMean, angleVar;
            angle.Gaussian.GetMeanAndVariance(out angleMean, out angleVar);
            double expVar = Math.Exp(-0.5 * angleVar);
            double mCos = Math.Cos(angleMean) * expVar;
            double mSin = Math.Sin(angleMean) * expVar;
            if (rotate.Dimension != 2)
                throw new ArgumentException("rotate.Dimension (" + rotate.Dimension + ") != 2");
            double prec = rotate.Precision[0, 0];
            if (rotate.Precision[0, 1] != 0)
                throw new ArgumentException("rotate.Precision is not diagonal");
            if (rotate.Precision[1, 1] != prec)
                throw new ArgumentException("rotate.Precision is not spherical");
#if false
			Vector rotateMean = rotate.GetMean();
			double mean = mCos*rotateMean[0] + mSin*rotateMean[1];
#else
            double rotateMean0 = rotate.MeanTimesPrecision[0] / rotate.Precision[0, 0];
            double rotateMean1 = rotate.MeanTimesPrecision[1] / rotate.Precision[1, 1];
            double mean = mCos * rotateMean0 + mSin * rotateMean1;
#endif
            if (double.IsNaN(mean))
                throw new ApplicationException("result is nan");
            return Gaussian.FromMeanAndPrecision(mean, prec);
        }

        /// <summary>VMP message to <c>y</c>.</summary>
        /// <param name="rotate">Incoming message from <c>rotate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="angle">Incoming message from <c>angle</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>y</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>y</c>. Because the factor is deterministic, <c>rotate</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(angle) p(angle) log(sum_rotate p(rotate) factor(rotate,x,y,angle)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rotate" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="angle" /> is not a proper distribution.</exception>
        public static Gaussian YAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] WrappedGaussian angle)
        {
            // for x ~ N(m,v):
            // E[cos(x)] = cos(m)*exp(-v/2)
            // E[sin(x)] = sin(m)*exp(-v/2)
            if (angle.Period != 2 * Math.PI)
                throw new ArgumentException("angle.Period (" + angle.Period + ") != 2*PI (" + 2 * Math.PI + ")");
            double angleMean, angleVar;
            angle.Gaussian.GetMeanAndVariance(out angleMean, out angleVar);
            double expVar = Math.Exp(-0.5 * angleVar);
            double mCos = Math.Cos(angleMean) * expVar;
            double mSin = Math.Sin(angleMean) * expVar;
            if (rotate.Dimension != 2)
                throw new ArgumentException("rotate.Dimension (" + rotate.Dimension + ") != 2");
            double prec = rotate.Precision[0, 0];
            if (rotate.Precision[0, 1] != 0)
                throw new ArgumentException("rotate.Precision is not diagonal");
            if (rotate.Precision[1, 1] != prec)
                throw new ArgumentException("rotate.Precision is not spherical");
#if false
			Vector rotateMean = rotate.GetMean();
			double mean = -mSin*rotateMean[0] + mCos*rotateMean[1];
#else
            double rotateMean0 = rotate.MeanTimesPrecision[0] / rotate.Precision[0, 0];
            double rotateMean1 = rotate.MeanTimesPrecision[1] / rotate.Precision[1, 1];
            double mean = -mSin * rotateMean0 + mCos * rotateMean1;
#endif
            return Gaussian.FromMeanAndPrecision(mean, prec);
        }

        /// <summary>VMP message to <c>x</c>.</summary>
        /// <param name="rotate">Incoming message from <c>rotate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="angle">Constant value for <c>angle</c>.</param>
        /// <returns>The outgoing VMP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>x</c> with <c>rotate</c> integrated out. The formula is <c>sum_rotate p(rotate) factor(rotate,x,y,angle)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rotate" /> is not a proper distribution.</exception>
        public static Gaussian XAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double angle)
        {
            return XAverageLogarithm(rotate, WrappedGaussian.PointMass(angle));
        }

        /// <summary>VMP message to <c>y</c>.</summary>
        /// <param name="rotate">Incoming message from <c>rotate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="angle">Constant value for <c>angle</c>.</param>
        /// <returns>The outgoing VMP message to the <c>y</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>y</c> with <c>rotate</c> integrated out. The formula is <c>sum_rotate p(rotate) factor(rotate,x,y,angle)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rotate" /> is not a proper distribution.</exception>
        public static Gaussian YAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double angle)
        {
            return YAverageLogarithm(rotate, WrappedGaussian.PointMass(angle));
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian RotateAverageLogarithmInit()
        {
            return new VectorGaussian(2);
        }

        /// <summary>VMP message to <c>rotate</c>.</summary>
        /// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="y">Incoming message from <c>y</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="angle">Incoming message from <c>angle</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>rotate</c> as the random arguments are varied. The formula is <c>proj[sum_(x,y,angle) p(x,y,angle) factor(rotate,x,y,angle)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="x" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="y" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="angle" /> is not a proper distribution.</exception>
        public static VectorGaussian RotateAverageLogarithm(
            [SkipIfUniform] Gaussian x, [SkipIfUniform] Gaussian y, [Proper] WrappedGaussian angle, VectorGaussian result)
        {
            // for x ~ N(m,v):
            // E[cos(x)] = cos(m)*exp(-v/2)
            // E[sin(x)] = sin(m)*exp(-v/2)
            if (angle.Period != 2 * Math.PI)
                throw new ArgumentException("angle.Period (" + angle.Period + ") != 2*PI (" + 2 * Math.PI + ")");
            double angleMean, angleVar;
            angle.Gaussian.GetMeanAndVariance(out angleMean, out angleVar);
            double expHalfVar = Math.Exp(-0.5 * angleVar);
            double mCos = Math.Cos(angleMean) * expHalfVar;
            double mSin = Math.Sin(angleMean) * expHalfVar;
            double mCos2 = mCos * mCos;
            double mSin2 = mSin * mSin;
            //  E[cos(x)^2] = 0.5 E[1+cos(2x)] = 0.5 (1 + cos(2m) exp(-2v))
            //  E[sin(x)^2] = E[1 - cos(x)^2] = 0.5 (1 - cos(2m) exp(-2v))
            double expVar = expHalfVar * expHalfVar;
            // cos2m = cos(2m)*exp(-v)
            double cos2m = 2 * mCos2 - expVar;
            double mCosSqr = 0.5 * (1 + cos2m * expVar);
            double mSinSqr = 1 - mCosSqr;
            double mSinCos = mSin * mCos * expVar;
            if (result.Dimension != 2)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") != 2");
            double mx, vx, my, vy;
            x.GetMeanAndVariance(out mx, out vx);
            y.GetMeanAndVariance(out my, out vy);
            Vector mean = Vector.Zero(2);
            mean[0] = mCos * mx - mSin * my;
            mean[1] = mSin * mx + mCos * my;
            double mx2 = mx * mx + vx;
            double my2 = my * my + vy;
            double mxy = mx * my;
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(2, 2);
            variance[0, 0] = mx2 * mCosSqr - 2 * mxy * mSinCos + my2 * mSinSqr - mean[0] * mean[0];
            variance[0, 1] = (mx2 - my2) * mSinCos + mxy * (mCosSqr - mSinSqr) - mean[0] * mean[1];
            variance[1, 0] = variance[0, 1];
            variance[1, 1] = mx2 * mSinSqr + 2 * mxy * mSinCos + my2 * mCosSqr - mean[1] * mean[1];
            result.SetMeanAndVariance(mean, variance);
            return result;
        }

        /// <summary>VMP message to <c>rotate</c>.</summary>
        /// <param name="x">Constant value for <c>x</c>.</param>
        /// <param name="y">Constant value for <c>y</c>.</param>
        /// <param name="angle">Incoming message from <c>angle</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>rotate</c> as the random arguments are varied. The formula is <c>proj[sum_(angle) p(angle) factor(rotate,x,y,angle)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="angle" /> is not a proper distribution.</exception>
        public static VectorGaussian RotateAverageLogarithm(double x, double y, [Proper] WrappedGaussian angle, VectorGaussian result)
        {
            return RotateAverageLogarithm(Gaussian.PointMass(x), Gaussian.PointMass(y), angle, result);
        }

        /// <summary>VMP message to <c>rotate</c>.</summary>
        /// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="y">Incoming message from <c>y</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="angle">Constant value for <c>angle</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>rotate</c> as the random arguments are varied. The formula is <c>proj[sum_(x,y) p(x,y) factor(rotate,x,y,angle)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="x" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="y" /> is not a proper distribution.</exception>
        public static VectorGaussian RotateAverageLogarithm([SkipIfUniform] Gaussian x, [SkipIfUniform] Gaussian y, double angle, VectorGaussian result)
        {
            return RotateAverageLogarithm(x, y, WrappedGaussian.PointMass(angle), result);
        }

#if false
		public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] Gaussian x, [Proper] Gaussian y, [Proper] WrappedGaussian angle)
		{
			return AngleAverageLogarithm(rotate, x.GetMean(), y.GetMean(), angle);
		}
		public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double x, double y, [Proper] WrappedGaussian angle)
		{
			if (rotate.Dimension != 2) throw new ArgumentException("rotate.Dimension ("+rotate.Dimension+") != 2");
			double rPrec = rotate.Precision[0, 0];
			if (rotate.Precision[0, 1] != 0) throw new ArgumentException("rotate.Precision is not diagonal");
			if (rotate.Precision[1, 1] != rPrec) throw new ArgumentException("rotate.Precision is not spherical");
			Vector rotateMean = rotate.GetMean();
			double a = x*rotateMean[0] + y*rotateMean[1];
			double b = x*rotateMean[1] - y*rotateMean[0];
			double c = Math.Sqrt(a*a + b*b)*rPrec;
			double angle0 = Math.Atan2(b, a);
			// the exact conditional is exp(c*cos(angle - angle0)) which is a von Mises distribution.
			// we will approximate this with a Gaussian lower bound that makes contact at the current angleMean.
			if (angle.Period != 2*Math.PI) throw new ArgumentException("angle.Period ("+angle.Period+") != 2*PI ("+2*Math.PI+")");
			double angleMean = angle.Gaussian.GetMean();
			double angleDiff = angleMean - angle0;
			double df = -c*Math.Sin(angleDiff);
			double precision = c*Math.Abs(Math.Cos(angleDiff*0.5)); // ensures a lower bound
			double meanTimesPrecision = angleMean*precision + df;
			if (double.IsNaN(meanTimesPrecision)) throw new ApplicationException("result is nan");
			WrappedGaussian result = WrappedGaussian.Uniform(angle.Period);
			result.Gaussian = Gaussian.FromNatural(meanTimesPrecision, precision);
			return result;
		}
#else
        /// <summary>VMP message to <c>angle</c>.</summary>
        /// <param name="rotate">Incoming message from <c>rotate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="y">Incoming message from <c>y</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>angle</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>angle</c>. Because the factor is deterministic, <c>rotate</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(x,y) p(x,y) log(sum_rotate p(rotate) factor(rotate,x,y,angle)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rotate" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="x" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="y" /> is not a proper distribution.</exception>
        public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] Gaussian x, [Proper] Gaussian y)
        {
            return AngleAverageLogarithm(rotate, x.GetMean(), y.GetMean());
        }

        /// <summary>VMP message to <c>angle</c>.</summary>
        /// <param name="rotate">Incoming message from <c>rotate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="x">Constant value for <c>x</c>.</param>
        /// <param name="y">Constant value for <c>y</c>.</param>
        /// <returns>The outgoing VMP message to the <c>angle</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>angle</c> with <c>rotate</c> integrated out. The formula is <c>sum_rotate p(rotate) factor(rotate,x,y,angle)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rotate" /> is not a proper distribution.</exception>
        public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double x, double y)
        {
            if (rotate.Dimension != 2)
                throw new ArgumentException("rotate.Dimension (" + rotate.Dimension + ") != 2");
            double rPrec = rotate.Precision[0, 0];
            if (rotate.Precision[0, 1] != 0)
                throw new ArgumentException("rotate.Precision is not diagonal");
            if (rotate.Precision[1, 1] != rPrec)
                throw new ArgumentException("rotate.Precision is not spherical");
#if false
			Vector rotateMean = rotate.GetMean();
			double a = x*rotateMean[0] + y*rotateMean[1];
			double b = x*rotateMean[1] - y*rotateMean[0]; 
#else
            double rotateMean0 = rotate.MeanTimesPrecision[0] / rotate.Precision[0, 0];
            double rotateMean1 = rotate.MeanTimesPrecision[1] / rotate.Precision[1, 1];
            double a = x * rotateMean0 + y * rotateMean1;
            double b = x * rotateMean1 - y * rotateMean0;
#endif
            double c = Math.Sqrt(a * a + b * b) * rPrec;
            double angle0 = Math.Atan2(b, a);
            // the exact conditional is exp(c*cos(angle - angle0)) which is a von Mises distribution.
            // we will approximate this with a Gaussian lower bound that makes contact at the mode.
            WrappedGaussian result = WrappedGaussian.Uniform();
            result.Gaussian = Gaussian.FromMeanAndPrecision(angle0, c);
            return result;
        }
#endif
    }
}
