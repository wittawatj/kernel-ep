using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;

// This file contains classes related to distributions.
namespace KernelEP.Tool{

	public abstract class KEPDist{
		public new static IKEPDist FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(className.Equals(DNormal.MATLAB_CLASS)){
				return DNormal.FromMatlabStruct(s);
			} else if(className.Equals(DBeta.MATLAB_CLASS)){
				return DBeta.FromMatlabStruct(s);
			} else{
				throw new ArgumentException("unknown distribution class.");
			}
		}
	}
	// marker interface for a distribution in this kernel EP framwork
	public interface IKEPDist{
		// return mean as a vector. Work on univariate distributions as well.
		Vector GetMeanVector();

		Matrix GetCovarianceMatrix();

		ISuff GetSuffStat();
	}
	// Interface for a distribution in kernel EP framework
	// Type D parametrizes the domain of the distribution e.g., double
	// Type V = type of variance. Either double or PositiveDefiniteMatrix
	//	public interface ParamDist<D, V> :  CanGetMean<D>,
	//	CanGetVariance<V>, KEPDist{
	//
	//		// Return true if this Dist is proper e.g., does not have
	//		// negative variance.
	////		public abstract bool IsProper();
	//
	//		// Get the dimension of the input to the distribution
	////		public abstract int GetDimension();
	//
	//	}

	// univariate distribution
	// This is just a wrapper. Infer.NET distributions are sealed (?).
	public interface IDistUni : IKEPDist{
		// Get the underlying wrapped Infer.NET distribution
		IDistribution<double> GetWrappedDistribution();

		double GetMean();

		double GetVariance();

	}

	// multivariate distribution
	// This is just a wrapper. Infer.NET distributions are sealed (?).
	public interface IDistMulti : IKEPDist{
		// Get the underlying wrapped Infer.NET distribution
		IDistribution<Vector> GetWrappedDistribution();
	}


	// A class representing sufficient statistic.
	// This is mainly used for construction of an outgoing message.
	// Sufficient statistic for a univariate distribution.
	//	public  class SuffStatUni : SuffStat<double, double>{
	//
	//		public double GetFirstMoment(){
	//			return -1;
	//		}
	//
	//		public double GetSecondMoment(){
	//			return -1;
	//		}
	//	}

	public interface ISuff{
		Vector GetFirstMomentAsVector();

		// uncentered 2nd moment
		PositiveDefiniteMatrix GetSecondMomentAsMatrix();

		// Return true if this is a sufficient statistic for a univariate
		// distribution i.e., first moment has one component, 2nd-moment matrix
		// has 1 entry.
		bool IsUnivariate();
	}

	public class SuffStat : ISuff{
		private readonly Vector firstMoment;
		private PositiveDefiniteMatrix secondMoment;

		public SuffStat(Vector firstMoment, PositiveDefiniteMatrix secondMoment){
			int c1 = firstMoment.Count;
			if(c1 * c1 != secondMoment.Count){
				throw new ArgumentException("1st & 2nd moments have incompatible dimensions.");
			}
			this.firstMoment = firstMoment;
			this.secondMoment = secondMoment;
		}

		public Vector GetFirstMomentAsVector(){
			return firstMoment;
		}

		// uncentered 2nd moment
		public PositiveDefiniteMatrix GetSecondMomentAsMatrix(){
			return secondMoment;
		}

		// Return true if this is a sufficient statistic for a univariate
		// distribution i.e., first moment has one component, 2nd-moment matrix
		// has 1 entry.
		public bool IsUnivariate(){
			return firstMoment.Count == 1;
		}
	}

	public abstract class DistBuilderBase{
		
		public static DistBuilderBase FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(className.Equals(DNormalBuilder.MATLAB_CLASS)){
				return DNormalBuilder.FromMatlabStruct(s);
			} else if(className.Equals(DBetaBuilder.MATLAB_CLASS)){
				return DBetaBuilder.FromMatlabStruct(s);
			} else if(className.Equals(DNormalLogVarBuilder.MATLAB_CLASS)){
				return DNormalLogVarBuilder.FromMatlabStruct(s);
			} else if(className.Equals(DBetaLogBuilder.MATLAB_CLASS)){
				return DBetaLogBuilder.FromMatlabStruct(s);
			} else{
				throw new ArgumentException("invalid MatlabStruct for a DistBuilder.");
			}

		}
	}
	// A builder for Dist objects from samples or list of statistics.
	public abstract class DistBuilder<T> : DistBuilderBase where T : IKEPDist{
		// Get the sufficient statistic for the distribution.
		// For example for a 1d Gaussian, the suff stat is [x; x.^2].
		// The returned suff stat s is such that fromStat(s) gives back d.
		//public abstract ISuff GetStat(T d);

		// Construct a distribution from the sufficient statistics s.
		public abstract T FromStat(ISuff s);

		// In Matlab code, sufficient statistic is represented with one vector.
		// For example, for a 1d Gaussian, s = [x, x^2].
		// This can be problematic for multivariate statistic as matrix needs
		// to be vectorized. If possible, ISuff should be used.
		public abstract T FromStat(Vector s);

		// Return a representaive statistic vector of the distribution
		public abstract Vector GetStat(T dist);
	}

	public class DNormalBuilder : DistBuilder<DNormal>{
		private static DNormalBuilder instance = null;
		public const string MATLAB_CLASS = "DistNormalBuilder";

		public static DNormalBuilder Instance{
			get{
				if(instance == null){
					instance = new DNormalBuilder();
				}
				return instance;
			}
		}
		// singleton pattern
		private DNormalBuilder(){
		}

		public override DNormal FromStat(ISuff s){
			double mean = s.GetFirstMomentAsVector()[0];
			double m2 = s.GetSecondMomentAsMatrix()[0, 0];
			double variance = m2 - mean * mean;
			return new DNormal(mean,variance);
		}

		public override DNormal FromStat(Vector s){
			// assume s = [x, x^2]
			if(s.Count != 2){
				throw new ArgumentException("2-dim vector for containing first 2 moments expected.");
			}
			double mean = s[0];
			double m2 = s[1];
			double variance = m2 - mean * mean;
			return new DNormal(mean,variance);
		}

		public override Vector GetStat(DNormal dist){
			// return mean and uncentered 2nd moment
			double mean = dist.GetMean();
			double m2 = dist.GetVariance() + mean*mean;
			return Vector.FromArray(new []{mean, m2});
		}

		public new static DNormalBuilder FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(DNormalBuilder));
			}
			return DNormalBuilder.Instance;
		}
	}

	// DistBuilder which represents a normal distribution with its mean and
	// log variance
	public class DNormalLogVarBuilder : DistBuilder<DNormal>{
		private static DNormalLogVarBuilder instance = null;
		public const string MATLAB_CLASS = "DNormalLogVarBuilder";

		public static DNormalLogVarBuilder Instance{
			get{
				if(instance == null){
					instance = new DNormalLogVarBuilder();
				}
				return instance;
			}
		}
		// singleton pattern
		private DNormalLogVarBuilder(){
		}

		public override DNormal FromStat(ISuff s){
			return DNormalBuilder.Instance.FromStat(s);
		}

		public override DNormal FromStat(Vector s){
			// assume s = [mean, log variance]
			if(s.Count != 2){
				throw new ArgumentException("2-dim vector for containing " +
				"[mean, log(variance)] expected.");
			}
			double mean = s[0];
			double logVar = s[1];
			double variance = Math.Exp(logVar);
			return new DNormal(mean,variance);
		}

		public override Vector GetStat(DNormal dist){
			double mean = dist.GetMean();
			double logVar = dist.GetVariance();
			return Vector.FromArray(new[]{ mean, logVar });
		}

		public new static DNormalLogVarBuilder FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(DNormalLogVarBuilder));
			}
			return DNormalLogVarBuilder.Instance;
		}
	}

	public class DBetaBuilder : DistBuilder<DBeta>{
		private static DBetaBuilder instance = null;
		public const string MATLAB_CLASS = "DistBetaBuilder";

		public static DBetaBuilder Instance{
			get{
				if(instance == null){
					instance = new DBetaBuilder();
				}
				return instance;

			}
		}
		// singleton pattern
		private DBetaBuilder(){
		}

		public override DBeta FromStat(ISuff s){
			double mean = s.GetFirstMomentAsVector()[0];
			double m2 = s.GetSecondMomentAsMatrix()[0, 0];
			double variance = m2 - mean * mean;
			Beta b = Beta.FromMeanAndVariance(mean, variance);
			return DBeta.FromBeta(b);
		}

		public override DBeta FromStat(Vector s){
			if(s.Count != 2){
				throw new ArgumentException("2-dim vector containing first 2 moments expected.");
			}
		
			double mean = s[0];
			// hack 
			double epsi = 1e-3;
			mean = Math.Max(Math.Min(mean, 1 - epsi), 0 + epsi);
			double m2 = s[1];
			double variance = m2 - mean * mean;
			// hack 
			if(variance < 0 || variance >= mean * (1 - mean)){
				variance = mean * (1 - mean) * 0.9;
			}
			Beta b = Beta.FromMeanAndVariance(mean, variance);
			return DBeta.FromBeta(b);
		}

		public override Vector GetStat(DBeta dist){
			// return mean and uncentered 2nd moment
			double mean = dist.GetMean();
			double m2 = dist.GetVariance() + mean*mean;
			return Vector.FromArray(new []{mean, m2}); 
		}

		public new static DBetaBuilder FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(DBetaBuilder));
			}
			return DBetaBuilder.Instance;
		}
	}

	// use log(alpha) and log(beta)
	public class DBetaLogBuilder : DistBuilder<DBeta>{
		private static DBetaLogBuilder instance = null;
		public const string MATLAB_CLASS = "DBetaLogBuilder";

		public static DBetaLogBuilder Instance{
			get{
				if(instance == null){
					instance = new DBetaLogBuilder();
				}
				return instance;

			}
		}
		// singleton pattern
		private DBetaLogBuilder(){
		}

		public override DBeta FromStat(ISuff s){
			double mean = s.GetFirstMomentAsVector()[0];
			double m2 = s.GetSecondMomentAsMatrix()[0, 0];
			double variance = m2 - mean * mean;
			Beta b = Beta.FromMeanAndVariance(mean, variance);
			return DBeta.FromBeta(b);
		}

		public override DBeta FromStat(Vector s){
			if(s.Count != 2){
				throw new ArgumentException("2-dim vector containing log(alpha), log(beta) expected.");
			}

			double alpha = Math.Exp(s[0]);
			double beta = Math.Exp(s[1]);
			Beta b = new Beta(alpha,beta);
			return DBeta.FromBeta(b);
		}

		public override Vector GetStat(DBeta dist){
			// stat = ( log(alpha), log(beta) )
			double la = Math.Log(dist.GetAlpha());
			double lb = Math.Log(dist.GetBeta());
			return Vector.FromArray(new[]{la, lb});
		}

		public new static DBetaLogBuilder FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(DBetaLogBuilder));
			}
			return DBetaLogBuilder.Instance;
		}
	}

	public class DVectorNormal : IDistMulti{
		private Vector mean;
		private Matrix covariance;

		public DVectorNormal(Vector mean, Matrix covariance){
			this.mean = mean;
			this.covariance = covariance;
		}

		public IDistribution<Vector> GetWrappedDistribution(){
			PositiveDefiniteMatrix pos = (PositiveDefiniteMatrix)covariance;
			return new VectorGaussian(mean,pos);
		}

		public Vector GetMeanVector(){
			return mean;
		}

		public Matrix GetCovarianceMatrix(){
			return covariance;
		}

		public ISuff GetSuffStat(){
			throw new NotImplementedException();
		}

	}

	public  class DNormal : IDistUni{
		private double mean;
		private double variance;
		public static string MATLAB_CLASS = "DistNormal";

		public DNormal(double mean, double variance){
			this.mean = mean;
			this.variance = variance;

		}

		public IDistribution<double> GetWrappedDistribution(){
			//MicrosoftResearch.Infer.Distributions.Gaussian
			return new Gaussian(this.mean,this.variance);
		}

		public  double GetMean(){
			return this.mean;
		}

		public  double GetVariance(){
			return this.variance;
		}

		public Vector GetMeanVector(){
			return Vector.FromArray(this.mean);
		}

		public Matrix GetCovarianceMatrix(){
			return PositiveDefiniteMatrix.IdentityScaledBy(1, this.variance);
		}

		public ISuff GetSuffStat(){
			Vector moment1 = GetMeanVector();
			double m2 = variance + mean * mean;
			PositiveDefiniteMatrix moment2 = PositiveDefiniteMatrix.IdentityScaledBy(1, m2);
			SuffStat s = new SuffStat(moment1,moment2);
			return s;
		}

		// construct from Infer.NET Gaussian object
		public static DNormal FromGaussian(Gaussian g){
			return new DNormal(g.GetMean(),g.GetVariance());
		}

		public static DNormal PointMass(double mean){
			// Construct a Gaussian with very small width.
			return DNormal.FromGaussian(Gaussian.PointMass(mean));
		}

		public override string ToString(){
			return String.Format("{0}(mean={1}, variance={2})", typeof(DNormal), 
				this.mean, this.variance);
		}

		public static DNormal FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(DNormal));
			}
			Vector meanVec = s.Get1DVector("mean");
			if(meanVec.Count != 1){
				throw new ArgumentException("mean vector is not 1 dimenion.");
			}
			double mean = meanVec[0];
			double variance = s.GetDouble("variance");
			return new DNormal(mean,variance);
		}
	}

	public class DBeta : IDistUni{

		private double alpha;
		private double beta;
		private readonly double mean;
		private readonly double variance;
		public static string MATLAB_CLASS = "DistBeta";

		public DBeta(double alpha, double beta){
			this.alpha = alpha;
			this.beta = beta; 
			GetDistribution().GetMeanAndVariance(out this.mean, out this.variance);
		}

		private Beta GetDistribution(){
			return new Beta(this.alpha,this.beta);
		}

		public IDistribution<double> GetWrappedDistribution(){
			return GetDistribution();
		}

		public double GetMean(){
			return this.mean;
		}

		public double GetVariance(){
			return this.variance;
		}
		public double GetAlpha(){
			return alpha;
		}
		public double GetBeta(){
			return beta;
		}

		public Vector GetMeanVector(){
			return Vector.FromArray(this.mean);
		}

		public Matrix GetCovarianceMatrix(){
			return PositiveDefiniteMatrix.IdentityScaledBy(1, this.variance);
		}

		public ISuff GetSuffStat(){
			Vector moment1 = GetMeanVector();
			double m2 = variance + mean * mean;
			PositiveDefiniteMatrix moment2 = PositiveDefiniteMatrix.IdentityScaledBy(1, m2);
			SuffStat s = new SuffStat(moment1,moment2);
			return s;
		}

		public override string ToString(){
			return String.Format("{0}(mean={1}, variance={2})", typeof(DBeta), 
				this.mean, this.variance);
		}

		// construct from Infer.NET Beta object
		public static DBeta FromBeta(Beta g){
			return new DBeta(g.TrueCount,g.FalseCount);
		}

		public static DBeta PointMass(double mean){
			return DBeta.FromBeta(Beta.PointMass(mean));
		}

		public static DBeta FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(DBeta));
			}
			double a = s.GetDouble("alpha");
			double b = s.GetDouble("beta");
			return new DBeta(a,b);
		}
	}

	public  class Uniform1DSampler : Sampleable<double>{

		private double lowerBound;
		private double upperBound;
		private Random random;

		public Uniform1DSampler(double lowerBound, double upperBound, Random random){
			if(lowerBound > upperBound){
				throw new ArgumentException("expect lowerBound <= upperBound");
			}
			this.random = random;
			this.lowerBound = lowerBound;
			this.upperBound = upperBound;
		}

		public Uniform1DSampler(double lowerBound, double upperBound) :
			this(lowerBound, upperBound, new Random()){

		}

		public double Sample(){
			double udraw = random.NextDouble();
			return udraw * (upperBound - lowerBound) + lowerBound;

		}

		public double Sample(double result){
			result = Sample();
			return result;
		}
	}
	//	// interface for a bunch of incoming messages
	//	// 2 incoming messages
	//	public abstract class InMsgs<A, B> : IIncomingMessages
	//	where A : KEPDist
	//	where B : KEPDist{
	//
	//		public abstract A GetMsg0();
	//
	//		public abstract B GetMsg1();
	//		// preserve the order of incoming messages.
	//		// If the factor is of the form p(x1 | x2, x3, ..), the order is always
	//		// x1, x2, x3, ....
	//		public abstract KEPDist[] AsArray();
	//	}
	//
	//	// 3 incoming messages
	//	public abstract class InMsgs<A, B, C> : IIncomingMessages{
	//
	//	}
	//	public interface IIncomingMessages{
	//		// number of incoming messages
	////		int IncomingCount();
	//
	//	}

}

