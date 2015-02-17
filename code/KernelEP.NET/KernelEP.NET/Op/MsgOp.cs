using System;
using System.Collections.Generic;
using System.Linq;
using KernelEP.Tool;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;

namespace KernelEP.Op{
	
	// class to control all operators in our framework
	public static class OpControl{
		// a map from a message operator type to its operator parameters
		private readonly static Dictionary<Type, OpParamsBase> opInternals
			= new Dictionary<Type, OpParamsBase>();

		static OpControl(){
			// default operator internals
//			opInternals.Add(KEPLogisticOp, some_thing )
		}

		public static void Add(Type t, OpParamsBase oi){
			if(oi == null){
				throw new ArgumentException("Operator internal cannot be null.");
			}
			opInternals.Add(t, oi);
		}

		public static OpParamsBase Get(Type t){
			if(!opInternals.ContainsKey(t)){
				throw new ArgumentException("Parameters undefined for messages operator: " + t);
			}
			return opInternals[t];
		}
	}

	// Equivalent class in Matlab code is FactorOperator.
	public abstract class OpParamsBase{

	}

	// Internal components (e.g., DistMapper to each variable) of a message
	// passing operator (e.g., XXXOp). This is useful in setting DistMapper for
	// each target variable to send to. One object of this class wraps all
	// DistMapper's.
	// A, B are types of the distributions.
	public class OpParams<A, B> : OpParamsBase 
	where A : IKEPDist where B : IKEPDist{
		protected DistMapper<A, A, B> DistMapper0;
		protected DistMapper<B, A, B> DistMapper1;

		public OpParams(DistMapper<A, A, B> dm0, DistMapper<B, A, B> dm1){
			this.DistMapper0 = dm0;
			this.DistMapper1 = dm1;

		}

		public DistMapper<A, A, B> GetDistMapper0(){
			return this.DistMapper0;
		}

		public DistMapper<B, A, B> GetDistMapper1(){
			return this.DistMapper1;
		}
	}

	public class LogisticOpParams : OpParams<DBeta, DNormal>{
		public LogisticOpParams(DistMapper<DBeta, DBeta, DNormal> dm0, 
		                        DistMapper<DNormal, DBeta, DNormal> dm1)
			: base(dm0, dm1){
			 
		}

		public static LogisticOpParams FromMatlabStruct(MatlabStruct s){
			// MatlabStruct is expected to represent a FactorOperator object 
			// from Matlab.
			// See files in saved/factor_op folder
			string className = s.GetString("className");
			if(!className.Equals("DefaultFactorOperator")){
				throw new ArgumentException("The input does not represent a FactorOperator.");
			}
			// 1d row cell array of DistMapper's => Object[]
			Object[,] mapperCell = s.GetCells("distMappers");
			if(mapperCell.Length != 2){
				// Expect exact 2 DistMapper's, one for Beta direction the
				// other for Gaussian.
				throw new ArgumentException("Loaded FactorOperator does not have 2 DistMapper's.");
			}

			// first one is for z (Beta) in p(z|x)
			MatlabStruct rawBetaMapper = new MatlabStruct((Dictionary<string, object>)mapperCell[0, 0]);
			DistMapper<DBeta, DBeta, DNormal> betaMapper = 
				GenericMapper<DBeta, DBeta, DNormal>.FromMatlabStruct(rawBetaMapper);

			// for x (Normal) in p(z|x)
			MatlabStruct rawNormalMapper = new MatlabStruct((Dictionary<string, object>)mapperCell[0, 1]);
			DistMapper<DNormal, DBeta, DNormal> normalMapper = 
				GenericMapper<DNormal, DBeta, DNormal>.FromMatlabStruct(rawNormalMapper);

			return new LogisticOpParams(betaMapper,normalMapper);
		}
	}
	//	public class OperatorInternal<A, B, C> : IOperatorInternal
	//	where A : KEPDist where B : KEPDist where C : KEPDist{
	//		// same structure as in OperatorInternal<A, B>
	//	}

	/// 
	/// <summary> EP message operator using importance sampler
	/// </summary>
	/// logistic = logistic(x)
	[FactorMethod(typeof(MMath),"Logistic",typeof(double))]
	[Quality(QualityBand.Experimental)]
	public static class ISLogisticOp{
		// Need at least 20000 or more.
		public static int GaussianImportanceSampleSize = 50000;
		public static int UniformImportanceSampleSize = 10000;

		public static Random UniformRandom = new Random(1);
		public static double UniformFrom = -20;
		public static double UniformTo = 20;

		//		public delegate double Proposal();
		//		public static Proposal proposal = DefaultProposal;
		public static Gaussian Proposal = Gaussian.FromMeanAndVariance(0, 200);
		//		public static Sampleable<double> Proposal = new Uniform1DSampler(-20, 20);

		public static Gaussian ProjToXGaussianProposal(Beta logistic, Gaussian x){
			// Get the projected message forming part of an outogoing message to X.
			double m1 = 0, m2 = 0;
			double wsum = 0;
			for(int i = 0; i < GaussianImportanceSampleSize; i++){
				double xi = Proposal.Sample();
				double yi = MMath.Logistic(xi);
				double lbi = logistic.GetLogProb(yi);
				double lgi = x.GetLogProb(xi);
				double si = Proposal.GetLogProb(xi);
				// importance weight
				double wi = Math.Exp(lbi + lgi - si);
				wsum += wi;
				m1 += wi * xi;
				m2 += wi * xi * xi;
			}
			m1 /= wsum;
			m2 /= wsum;
			double projM = m1;
			double projV = m2 - m1 * m1;
			return Gaussian.FromMeanAndVariance(projM, projV);
		}

		public static Beta ProjToLogisticGaussianProposal(Beta logistic, Gaussian x){
			// Get the projected message forming part of an outgoing message to logistic.

			double m1 = 0, m2 = 0;
			double wsum = 0;
			for(int i = 0; i < GaussianImportanceSampleSize; i++){
				double xi = Proposal.Sample();
				double yi = MMath.Logistic(xi);
				double lbi = logistic.GetLogProb(yi);
				double lgi = x.GetLogProb(xi);
				double si = Proposal.GetLogProb(xi);
				// importance weight
				double wi = Math.Exp(lbi + lgi - si);
				wsum += wi;
				m1 += wi * yi;
				// uncentered second moment
				m2 += wi * yi * yi;
			}
			m1 /= wsum;
			m2 /= wsum;
			double projM = m1;
			double projV = m2 - m1 * m1;
			// method of moments http://en.wikipedia.org/wiki/Beta_distribution#Method_of_moments
			if( (Math.Abs(1 - projM) < 1e-6 || Math.Abs(projM) < 1e-6)
			   && Math.Abs(projV) < 1e-6){
				Console.WriteLine("!! neg variance in ProjToLogisticGaussianProposal: m={0}, v={1}", projM, projV);
				// mean very close to 0 or 1. Tiny variance. 
				// Ideally we want to make a point mass at 1.
				// But a point mass will not be a good taget for a regression function 
				// to learn.
				projM = Math.Abs(projM);
				// This should overestimate the variance of proj.
				// Overestimating should be better than underestimating.
				projV = projM*(1-projM);
			}

			return Beta.FromMeanAndVariance(projM, projV);
		}

		public static Gaussian XAverageConditional(Beta logistic, Gaussian x){
			Console.WriteLine("Direction X: beta: {0} , gaussian: {1}", logistic, x);
			
			// initial toX is improper
			Gaussian projToX = ProjToXGaussianProposal(logistic, x);
//			Gaussian projToX = ProjToXUniformProposal(logistic, x);
			double projM, projV;
			projToX.GetMeanAndVariance(out projM, out projV);
			Gaussian toX = Gaussian.FromNatural(1, -1);
			toX.SetToRatio(projToX, x, true);
			// The following is my own hack
//			double varReduceFactor = 1;
//			while(!toX.IsProper()){
//				Gaussian proj = Gaussian.FromMeanAndVariance(projM, projV / varReduceFactor);
//				toX = proj / x;
//				// increase precision so that division gives a proper outgoing message.
//				varReduceFactor *= 2;
//			}
			Console.WriteLine("Proj To X: Gaussian({0}, {1})", projToX.GetMean(), 
				projToX.GetVariance());
			Console.WriteLine("To X: Gaussian({0}, {1})", toX.GetMean(), toX.GetVariance());
			Console.WriteLine();
			return toX;
		}



		/// <summary>
		/// EP message to 'logistic' with importance sampling.
		/// </summary>
		public static Beta LogisticAverageConditional(Beta logistic, Gaussian x){
			Console.WriteLine("Direction logistic: beta: {0} , gaussian: {1}", logistic, x);
			Beta projToLogistic = ProjToLogisticGaussianProposal(logistic, x);
			//			Beta projToLogistic = ProjToLogisticUniformProposal(logistic, x);
			Beta toL = Beta.FromMeanLogs(Math.Log(0.5), 0.2);
			toL.SetToRatio(projToLogistic, logistic);
			// initial toL is improper
			//			Beta toL = Beta.FromMeanAndVariance(0.5, -0.5);

			//			double varReduceFactor = 1;
			//			while(!toL.IsProper()){
			//				Beta proj = Beta.FromMeanAndVariance(projM, projV / varReduceFactor);
			//				toL = proj / logistic;
			//				// increase precision so that division gives a proper outgoing message.
			//				varReduceFactor *= 2;
			//			}
			Console.WriteLine("Proj To Logistic: {0}", projToLogistic);
			Console.WriteLine("To logistic: {0}", toL);
			Console.WriteLine();
			return toL;
		}

		public static Gaussian ProjToXUniformProposal(Beta logistic, Gaussian x){
			// Get the projected message forming part of an outogoing message to X.
			double m1 = 0, m2 = 0;
			double wsum = 0;
			double urange = UniformTo - UniformFrom;
			double si = -Math.Log(urange);
			var grid = Enumerable.Range(0, UniformImportanceSampleSize)
				.Select(i => urange * i / (double)(UniformImportanceSampleSize - 1) + UniformFrom);

			double[] xlocations = grid.ToArray<double>();
			for(int i = 0; i < UniformImportanceSampleSize; i++){
				//				double xi = UniformRandom.NextDouble() * urange + UniformFrom;
				double xi = xlocations[i];
				//				Console.WriteLine("xi: {0}", xi);
				double yi = MMath.Logistic(xi);
				double lbi = logistic.GetLogProb(yi);
				double lgi = x.GetLogProb(xi);

				// importance weight
				double wi = Math.Exp(lbi + lgi - si);
				wsum += wi;
				m1 += wi * xi;
				m2 += wi * xi * xi;
			}
			m1 /= wsum;
			m2 /= wsum;
			double projM = m1;
			double projV = m2 - m1 * m1;
			return Gaussian.FromMeanAndVariance(projM, projV);
		}


	

		public static Beta ProjToLogisticUniformProposal(Beta logistic, Gaussian x){
			// Get the projected message forming part of an outgoing message to logistic.

			double m1 = 0, m2 = 0;
			double wsum = 0;
			double urange = UniformTo - UniformFrom;
			double si = -Math.Log(urange);
			var grid = Enumerable.Range(0, UniformImportanceSampleSize)
				.Select(i => urange * i / (double)(UniformImportanceSampleSize - 1) + UniformFrom);
			double[] xlocations = grid.ToArray<double>();

			for(int i = 0; i < UniformImportanceSampleSize; i++){
				//				double xi = UniformRandom.NextDouble() * urange + UniformFrom;
				double xi = xlocations[i];
				double yi = MMath.Logistic(xi);

				double lbi = logistic.GetLogProb(yi);
				double lgi = x.GetLogProb(xi);

				// importance weight
				double wi = Math.Exp(lbi + lgi - si);
				//				double wi = Math.Exp(lbi + lgi);
				wsum += wi;
				m1 += wi * yi;
				m2 += wi * yi * yi;
			}
			m1 /= wsum;
			m2 /= wsum;
			double projM = m1;
			double projV = m2 - m1 * m1;
			return Beta.FromMeanAndVariance(projM, projV);
		}
	}

	/// 
	/// <summary> Provide an EP operator for logistic (scalar sigmoid) factor.
	/// </summary>
	/// logistic = logistic(x)
	[FactorMethod(typeof(MMath),"Logistic",typeof(double))]
	[Quality(QualityBand.Experimental)]
	public static class KEPLogisticOp{
		// This is used when calling the true Infer.net message operator.
		private static Gaussian falseMsg = LogisticOp2.FalseMsgInit();
		// If true, print true messages sent by an Infer.NET's implementation of
		// logistic factor message operator.
		private static bool isPrintTrueMessages = false;

		// static constructor called automatically only once before other
		// static methods.
		static KEPLogisticOp(){

		}

		public static void SetPrintTrueMessages(bool v){
			LogisticOp2.IsPrintLog = !v;
			isPrintTrueMessages = v;
		}

		private static OpParams<DBeta, DNormal> GetOpParams(){
			OpParamsBase oi = OpControl.Get(typeof(KEPLogisticOp));
			OpParams<DBeta, DNormal> oi1 = (OpParams<DBeta, DNormal>)oi;
			return oi1;

		}

		private static DistMapper<DBeta, DBeta, DNormal> ToLogisticMapper(){
			OpParams<DBeta, DNormal> p = GetOpParams();
			return p.GetDistMapper0();
		}

		private static DistMapper<DNormal, DBeta, DNormal> ToXMapper(){
			OpParams<DBeta, DNormal> p = GetOpParams();
			return p.GetDistMapper1();
		}

		/// <summary>
		/// EP message to 'x'
		/// </summary>
		/// <param name="logistic">Constant value for 'logistic'.</param>
		/// <returns>The outgoing EP message to the 'x' argument</returns>
		//		public static Gaussian XAverageConditional(double logistic){
		//			return Gaussian.PointMass(MMath.Logit(logistic));
		//		}

		public static Gaussian XAverageConditional(Beta logistic, Gaussian x){


			Console.WriteLine("XAverageConditional. From: beta: {0} , gaussian: {1}", logistic, x);
			DistMapper<DNormal, DBeta, DNormal> dm = ToXMapper();
			DBeta l = DBeta.FromBeta(logistic);
//			if( Math.Abs(logistic.GetMean() - 1.0/3) < 1e-6){
//				// assume point mass at 0
//				l = DBeta.FromBeta(Beta.FromMeanAndVariance(0.03, 0.01));
//			}else if(Math.Abs(logistic.GetMean() - 2.0/3) < 1e-6){
//				// assume point mass at 1
//				l = DBeta.FromBeta(Beta.FromMeanAndVariance(0.97, 0.01));
//			}
			DNormal fromx = DNormal.FromGaussian(x);
			DNormal toXProjected = dm.MapToDist(l, fromx);
			Gaussian projX = (Gaussian)toXProjected.GetWrappedDistribution();
			Console.WriteLine("toX: {0}", projX);
			if(isPrintTrueMessages){
				falseMsg = LogisticOp2.FalseMsg(logistic, x, falseMsg);
				Gaussian trueToX = LogisticOp2.XAverageConditional(logistic, x, falseMsg);
				Console.WriteLine("True to x: {0}", trueToX);
			}
			return projX;

			Gaussian toX = projX / x;
			if(!toX.IsProper()){
				// According to Tom Minka, Gaussian times sigmoid likelihood (proj distribution)
				// will give a lower variance than the original Gaussian. 
				// That means proj dist / Gaussian will have positive precision.
				// So the outgoing message from here will always be proper.
				// force proper. If improper, getting mean will throw an exception.
				// This should never happen
				double projMTP, projP;
				projX.GetNatural(out projMTP, out projP);
				double xMTP, xP;
				x.GetNatural(out xMTP, out xP);
				double projMean = projMTP / projP;
				double newProjP = xP + 1.0 / 1e5;
				projX = Gaussian.FromNatural(projMean * newProjP, newProjP);
				// Make sure that the precision of projX > x, so that 
				// projX / x is proper. 
				Gaussian fixedToX = projX / x;
				Console.WriteLine("toX improper: {0}. fixed toX: {1}", toX, fixedToX);
				toX = fixedToX;

			} else{
				Console.WriteLine("toX: {0}", toX);
			}


			Console.WriteLine();

			// skipping EP
//			if(!toX.IsProper()){
//				return Gaussian.Uniform();
//			}
			return toX;
		}

		/// <summary>
		/// EP message to 'logistic'
		/// </summary>
		/// <param name="logistic">Incoming message from 'logistic'.</param>
		/// <param name="x">Incoming message from 'x'. </param>
		/// <returns>The outgoing EP message to the 'logistic' argument</returns>
		public static Beta LogisticAverageConditional(Beta logistic, Gaussian x){
//			if(logistic.IsPointMass){
//				Console.WriteLine("beta point mass");
//			}
			DBeta fromL = DBeta.FromBeta(logistic);
			Console.WriteLine("LogisticAverageConditional. beta: {0} , gaussian: {1}", logistic, x);
			DistMapper<DBeta, DBeta, DNormal> dm = ToLogisticMapper();

			DNormal fromX = DNormal.FromGaussian(x);
			DBeta toLProjected = dm.MapToDist(fromL, fromX);
			Beta projL = (Beta)toLProjected.GetWrappedDistribution();
			Console.WriteLine("toL: {0}", projL);
			if(isPrintTrueMessages){
				Beta trueToLogistic = LogisticOp2.LogisticAverageConditional(logistic, x, falseMsg);
				Console.WriteLine("True to logistic: {0}", trueToLogistic);
			}
			return projL;
			Beta toL = projL / logistic;
			
//			double projT = projL.TrueCount;
//			double projF = projL.FalseCount;
//			double loT = logistic.TrueCount;
//			double loF = logistic.FalseCount;

			if(!projL.IsProper()){
//				double projMean = projL.GetMean();
//				double projVar = projL.GetVariance();

			}
			if(!toL.IsProper()){
				Console.WriteLine("toL improper: {0}", toL);
				// force proper. If improper, getting mean will throw an exception.

				// proper if projT > loT - 1 and projF > loF - 1
//				toL = Beta.Uniform();
			} else{
				Console.WriteLine("toL: {0}", toL);
			}

			Console.WriteLine();
			return toL;
		}

		// When logistic value is observed
		//		public static Gaussian XAverageConditional(double logistic, Gaussian x){
		//			Console.WriteLine("constant logistic: {0}", logistic);
		//			DistMapper<DNormal, DBeta, DNormal> dm = ToXMapper();
		//			// construct a Beta with mean at logistic with very small variance
		//			double variance = 0.01;
		//			DBeta l = DBeta.FromBeta(Beta.FromMeanAndVariance(logistic, variance));
		//			DNormal fromx = DNormal.FromGaussian(x);
		//			DNormal toXProjected = dm.MapToDist(l, fromx);
		//			Gaussian toX = (Gaussian)toXProjected.GetWrappedDistribution() / x;
		//			return toX;
		//		}
	

		//		public static Beta LogisticAverageConditional(double logistic, Gaussian x){
		//			Console.WriteLine("constant logistic to logistic: {0}", logistic);
		////			if(x.IsPointMass)
		////				return Beta.PointMass(MMath.Logistic(x.Point));
		//			DistMapper<DBeta, DBeta, DNormal> dm = ToLogisticMapper();
		//
		//			// construct a Beta with mean at logistic with very small variance
		//			double variance = 0.01;
		//			DBeta fromL = DBeta.FromBeta(Beta.FromMeanAndVariance(logistic, variance));
		//			DNormal fromX = DNormal.FromGaussian(x);
		//			DBeta toLProjected = dm.MapToDist(fromL, fromX);
		//			Beta toL = (Beta)toLProjected.GetWrappedDistribution() / (Beta)fromL.GetWrappedDistribution();
		//			return toL;
		//		}

	}
	//end of KEPLogisticOp class


	// Basically a copy of the original LogisticOp in Infer.NET with print
	// statements.
	/**
	 * When running a binary logistic regression model with 0-1 labels observed,
	 * the main method for sending EP messages to X (argument of Logistic) is 
	 * XAverageConditional([SkipIfUniform] Beta logistic, Gaussian falseMsg).
	 * 
	*/
	/// <summary>Provides outgoing messages for <see cref="MMath.Logistic(double)" />, given random arguments to the function.</summary>
	[FactorMethod(typeof(MMath),"Logistic",typeof(double))]
	[Quality(QualityBand.Stable)]
	[Buffers("falseMsg")]
	public static class LogisticOp2{
		// true to save all messages from/to X
		public static bool IsCollectXMessages = true;
		// true to save all messages from/to X
		public static bool IsCollectLogisticMessages = true;
		// If true, collect projected messages instead of the outgoing messages.
		// Outgoing messages can be obtained by dividing these messages by the cavity.
		public static bool IsCollectProjMsgs = true;
		public static bool IsPrintLog = true;

		// A list of Tuple<Gaussian, Gaussian, Beta> for a Gaussian outgoing
		// message, incoming x message, incoming logistic message.
		private static List<Tuple<Gaussian, Gaussian, Beta>> toXMessages;
		// A list of Tuple<Gaussian, Gaussian, Beta> for a Beta outgoing
		// message, incoming x message, incoming logistic message.
		private static List<Tuple<Beta, Gaussian, Beta>> toLogisticMessages;

		// static constructor
		static LogisticOp2(){
			ResetMessageCollection();
		}
		//--------- methods for collecting messages --------
		public static List<Tuple<Gaussian, Gaussian, Beta>> GetToXMessages(){
			return toXMessages;
		}

		public static List<Tuple<Beta, Gaussian, Beta>> GetToLogisticMessages(){
			return toLogisticMessages;
		}

		/**
		 * Reset all message collection data structure.
		*/
		public static void  ResetMessageCollection(){
			toXMessages = new List<Tuple<Gaussian, Gaussian, Beta>>();
			toLogisticMessages = new List<Tuple<Beta, Gaussian, Beta>>();
		}

		//----------------------------------

		/// <summary>EP message to <c>x</c>.</summary>
		/// <param name="logistic">Incoming message from <c>logistic</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <param name="falseMsg">Buffer <c>falseMsg</c>.</param>
		/// <returns>The outgoing EP message to the <c>x</c> argument.</returns>
		/// <remarks>
		///   <para>The outgoing message is a distribution matching the moments of <c>x</c> as the random arguments are varied. The formula is <c>proj[p(x) sum_(logistic) p(logistic) factor(logistic,x)]/p(x)</c>.</para>
		/// </remarks>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="logistic" /> is not a proper distribution.</exception>
		public static Gaussian XAverageConditional([SkipIfUniform] Beta logistic, 
		                                           Gaussian x, Gaussian falseMsg){

			if(IsPrintLog){
				Console.WriteLine("XAverageConditional([SkipIfUniform] Beta logistic, Gaussian falseMsg)");
				Console.WriteLine("x: {0}", x);
			}

			if(logistic.IsPointMass)
				return XAverageConditional(logistic.Point);
			if(falseMsg.IsPointMass)
				throw new ArgumentException("falseMsg is a point mass");
			// sigma(x)^(a-1) sigma(-x)^(b-1)
			// = e^((a-1)x) falseMsg^(a+b-2)
			// e^((a-1)x) = Gaussian.FromNatural(a-1,0)
			double tc1 = logistic.TrueCount - 1;
			double fc1 = logistic.FalseCount - 1;
			Gaussian toX = Gaussian.FromNatural((tc1 + fc1) * falseMsg.MeanTimesPrecision + tc1, (tc1 + fc1) * falseMsg.Precision);

			// ### Message collection ###
			if(IsCollectXMessages){
				if(IsCollectProjMsgs){
					// compute proj message. This can be expensive.
					Gaussian projX = ISLogisticOp.ProjToXGaussianProposal(logistic, x);
					var pair = new Tuple<Gaussian, Gaussian, Beta>(projX,x,logistic);
					toXMessages.Add(pair);
				} else{
					// collect outgoing message
					var pair = new Tuple<Gaussian, Gaussian, Beta>(toX,x,logistic);
					toXMessages.Add(pair);
				}
			}
			return toX;
		}

		/// <summary>EP message to <c>logistic</c>.</summary>
		/// <param name="logistic">Incoming message from <c>logistic</c>.</param>
		/// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <param name="falseMsg">Buffer <c>falseMsg</c>.</param>
		/// <returns>The outgoing EP message to the <c>logistic</c> argument.</returns>
		/// <remarks>
		///   <para>The outgoing message is a distribution matching the moments of <c>logistic</c> as the random arguments are varied. The formula is <c>proj[p(logistic) sum_(x) p(x) factor(logistic,x)]/p(logistic)</c>.</para>
		/// </remarks>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="x" /> is not a proper distribution.</exception>
		public static Beta LogisticAverageConditional(Beta logistic, 
		                                              [Proper] Gaussian x, Gaussian falseMsg){
			if(IsPrintLog){
				Console.WriteLine("LogisticAverageConditional(Beta logistic, [Proper] Gaussian x, Gaussian falseMsg)");
				Console.WriteLine("Gaussian x: " + x);
			}

			Beta toLogistic;

			if(x.IsPointMass){
				toLogistic = Beta.PointMass(MMath.Logistic(x.Point));
				CollectCheckLogisticMsg(toLogistic, x, logistic);
				return toLogistic;
			}
				

			if(logistic.IsPointMass || x.IsUniform()){
				toLogistic = Beta.Uniform();
				CollectCheckLogisticMsg(toLogistic, x, logistic);
				return toLogistic;
			}

			double m, v;
			x.GetMeanAndVariance(out m, out v);
		
			// For observed Bernoulli variable, logistic will be Beta(1, 2) or 
			// Beta(2, 1)
			if((logistic.TrueCount == 2 && logistic.FalseCount == 1) ||
			   (logistic.TrueCount == 1 && logistic.FalseCount == 2) ||
			   logistic.IsUniform()){
				// shortcut for the common case
				// result is a Beta distribution satisfying:
				// int_p to_p(p) p dp = int_x sigma(x) qnoti(x) dx
				// int_p to_p(p) p^2 dp = int_x sigma(x)^2 qnoti(x) dx
				// the second constraint can be rewritten as:
				// int_p to_p(p) p (1-p) dp = int_x sigma(x) (1 - sigma(x)) qnoti(x) dx
				// the constraints are the same if we replace p with (1-p)
				double mean = MMath.LogisticGaussian(m, v);
				// meanTF = E[p] - E[p^2]
				double meanTF = MMath.LogisticGaussianDerivative(m, v);
				double meanSquare = mean - meanTF;
				double toLogisticVar = meanSquare - mean * mean;
				// Wittawat added this
				if(toLogisticVar < 0){
					Console.WriteLine("<0");
					toLogisticVar = 0;
				}
				toLogistic = Beta.FromMeanAndVariance(mean, toLogisticVar);
			} else{
				// stabilized EP message
				// choose a normalized distribution to_p such that:
				// int_p to_p(p) qnoti(p) dp = int_x qnoti(sigma(x)) qnoti(x) dx
				// int_p to_p(p) p qnoti(p) dp = int_x qnoti(sigma(x)) sigma(x) qnoti(x) dx
				double logZ = LogAverageFactor(logistic, x, falseMsg) + logistic.GetLogNormalizer(); // log int_x logistic(sigma(x)) N(x;m,v) dx
				Gaussian post = XAverageConditional(logistic, x, falseMsg) * x;
				double mp, vp;
				post.GetMeanAndVariance(out mp, out vp);
				double tc1 = logistic.TrueCount - 1;
				double fc1 = logistic.FalseCount - 1;
				double Ep;
				if(tc1 + fc1 == 0){
					Beta logistic1 = new Beta(logistic.TrueCount + 1,logistic.FalseCount);
					double logZp = LogAverageFactor(logistic1, x, falseMsg) + logistic1.GetLogNormalizer();
					Ep = Math.Exp(logZp - logZ);
				} else{
					// Ep = int_p to_p(p) p qnoti(p) dp / int_p to_p(p) qnoti(p) dp
					// mp = m + v (a - (a+b) Ep)
					Ep = (tc1 - (mp - m) / v) / (tc1 + fc1);
				}

				toLogistic = BetaFromMeanAndIntegral(Ep, logZ, tc1, fc1);

			}
			CollectCheckLogisticMsg(toLogistic, x, logistic);
			return toLogistic;
		}

		private static void CollectCheckLogisticMsg(Beta toLogistic, Gaussian x,
		                                            Beta logistic){
			// ### Message collection ###
			if(IsCollectLogisticMessages){

				if(IsCollectProjMsgs){
					// compute proj message. This can be expensive.
					Beta projLogistic = ISLogisticOp.ProjToLogisticGaussianProposal(logistic, x);
					var pair = new Tuple<Beta, Gaussian, Beta>(projLogistic,x,logistic);
					toLogisticMessages.Add(pair);
				} else{
					// collect outgoing message
					var pair = new Tuple<Beta, Gaussian, Beta>(toLogistic,x,logistic);
					toLogisticMessages.Add(pair);
				}

			}
		}

		/// <summary>Evidence message for EP.</summary>
		/// <param name="logistic">Constant value for <c>logistic</c>.</param>
		/// <param name="x">Constant value for <c>x</c>.</param>
		/// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(factor(logistic,x))</c>.</para>
		/// </remarks>
		public static double LogAverageFactor(double logistic, double x){
			Console.WriteLine("LogAverageFactor(double logistic, double x)");
			return (logistic == MMath.Logistic(x)) ? 0.0 : Double.NegativeInfinity;
		}

		/// <summary>Evidence message for EP.</summary>
		/// <param name="logistic">Constant value for <c>logistic</c>.</param>
		/// <param name="x">Constant value for <c>x</c>.</param>
		/// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(factor(logistic,x))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
		/// </remarks>
		public static double LogEvidenceRatio(double logistic, double x){
			Console.WriteLine("LogEvidenceRatio(double logistic, double x)");
			return LogAverageFactor(logistic, x);
		}

		/// <summary>Evidence message for VMP.</summary>
		/// <param name="logistic">Constant value for <c>logistic</c>.</param>
		/// <param name="x">Constant value for <c>x</c>.</param>
		/// <returns>Zero.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(factor(logistic,x))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
		/// </remarks>
		public static double AverageLogFactor(double logistic, double x){
			Console.WriteLine("AverageLogFactor(double logistic, double x)");
			return LogAverageFactor(logistic, x);
		}

		/// <summary>Evidence message for EP.</summary>
		/// <param name="logistic">Incoming message from <c>logistic</c>.</param>
		/// <param name="x">Constant value for <c>x</c>.</param>
		/// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(sum_(logistic) p(logistic) factor(logistic,x))</c>.</para>
		/// </remarks>
		public static double LogAverageFactor(Beta logistic, double x){
			Console.WriteLine("LogAverageFactor(Beta logistic, double x)");
			return logistic.GetLogProb(MMath.Logistic(x));
		}

		/// <summary>Evidence message for EP.</summary>
		/// <param name="logistic">Constant value for <c>logistic</c>.</param>
		/// <param name="x">Incoming message from <c>x</c>.</param>
		/// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(sum_(x) p(x) factor(logistic,x))</c>.</para>
		/// </remarks>
		public static double LogAverageFactor(double logistic, Gaussian x){
			Console.WriteLine("LogAverageFactor(double logistic, Gaussian x)");
			if(logistic >= 1.0 || logistic <= 0.0)
				return x.GetLogProb(MMath.Logit(logistic));
			// p(y,x) = delta(y - 1/(1+exp(-x))) N(x;mx,vx)
			// x = log(y/(1-y))
			// dx = 1/(y*(1-y))
			return x.GetLogProb(MMath.Logit(logistic)) / (logistic * (1 - logistic));
		}

		/// <summary>Evidence message for EP.</summary>
		/// <param name="logistic">Constant value for <c>logistic</c>.</param>
		/// <param name="x">Incoming message from <c>x</c>.</param>
		/// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(sum_(x) p(x) factor(logistic,x))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
		/// </remarks>
		public static double LogEvidenceRatio(double logistic, Gaussian x){
			Console.WriteLine("LogEvidenceRatio(double logistic, Gaussian x)");
			return LogAverageFactor(logistic, x);
		}

		/// <summary>Evidence message for EP.</summary>
		/// <param name="logistic">Incoming message from <c>logistic</c>.</param>
		/// <param name="x">Incoming message from <c>x</c>.</param>
		/// <param name="falseMsg">Buffer <c>falseMsg</c>.</param>
		/// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(sum_(logistic,x) p(logistic,x) factor(logistic,x))</c>.</para>
		/// </remarks>
		public static double LogAverageFactor(Beta logistic, Gaussian x, Gaussian falseMsg){
			Console.WriteLine("LogAverageFactor(Beta logistic, Gaussian x, Gaussian falseMsg)");
			// return log(int_y int_x delta(y - Logistic(x)) Beta(y) Gaussian(x) dx dy)
			double m, v;
			x.GetMeanAndVariance(out m, out v);
			if(logistic.TrueCount == 2 && logistic.FalseCount == 1){
				// shortcut for common case
				return Math.Log(2 * MMath.LogisticGaussian(m, v));
			} else if(logistic.TrueCount == 1 && logistic.FalseCount == 2){
				return Math.Log(2 * MMath.LogisticGaussian(-m, v));
			} else{
				// logistic(sigma(x)) N(x;m,v)
				// = sigma(x)^(a-1) sigma(-x)^(b-1) N(x;m,v) gamma(a+b)/gamma(a)/gamma(b)
				// = e^((a-1)x) sigma(-x)^(a+b-2) N(x;m,v)
				// = sigma(-x)^(a+b-2) N(x;m+(a-1)v,v) exp((a-1)m + (a-1)^2 v/2)
				// int_x logistic(sigma(x)) N(x;m,v) dx 
				// =approx (int_x sigma(-x)/falseMsg(x) falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(a+b-2) 
				//       * (int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(1 - (a+b-2))
				//       *  exp((a-1)m + (a-1)^2 v/2) gamma(a+b)/gamma(a)/gamma(b)
				// This formula comes from (66) in Minka (2005)
				// Alternatively,
				// =approx (int_x falseMsg(x)/sigma(-x) falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(-(a+b-2))
				//       * (int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(1 + (a+b-2))
				//       *  exp((a-1)m + (a-1)^2 v/2) gamma(a+b)/gamma(a)/gamma(b)
				double tc1 = logistic.TrueCount - 1;
				double fc1 = logistic.FalseCount - 1;
				Gaussian prior = new Gaussian(m + tc1 * v,v);
				if(tc1 + fc1 < 0){
					// numerator2 = int_x falseMsg(x)^(a+b-1) N(x;m+(a-1)v,v) dx
					double numerator2 = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1 + 1);
					Gaussian prior2 = prior * (falseMsg ^ (tc1 + fc1 + 1));
					double mp, vp;
					prior2.GetMeanAndVariance(out mp, out vp);
					// numerator = int_x (1+exp(x)) falseMsg(x)^(a+b-1) N(x;m+(a-1)v,v) dx / int_x falseMsg(x)^(a+b-1) N(x;m+(a-1)v,v) dx
					double numerator = Math.Log(1 + Math.Exp(mp + 0.5 * vp));
					// denominator = int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v) dx
					double denominator = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1);
					return -(tc1 + fc1) * (numerator + numerator2 - denominator) + denominator + (tc1 * m + tc1 * tc1 * v * 0.5) - logistic.GetLogNormalizer();
				} else{
					// numerator2 = int_x falseMsg(x)^(a+b-3) N(x;m+(a-1)v,v) dx
					double numerator2 = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1 - 1);
					Gaussian prior2 = prior * (falseMsg ^ (tc1 + fc1 - 1));
					double mp, vp;
					prior2.GetMeanAndVariance(out mp, out vp);
					// numerator = int_x sigma(-x) falseMsg(x)^(a+b-3) N(x;m+(a-1)v,v) dx / int_x falseMsg(x)^(a+b-3) N(x;m+(a-1)v,v) dx
					double numerator = Math.Log(MMath.LogisticGaussian(-mp, vp));
					// denominator = int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v) dx
					double denominator = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1);
					return (tc1 + fc1) * (numerator + numerator2 - denominator) + denominator + (tc1 * m + tc1 * tc1 * v * 0.5) - logistic.GetLogNormalizer();
				}
			}
		}

		/// <summary>Evidence message for EP.</summary>
		/// <param name="logistic">Incoming message from <c>logistic</c>.</param>
		/// <param name="x">Incoming message from <c>x</c>.</param>
		/// <param name="falseMsg">Buffer <c>falseMsg</c>.</param>
		/// <param name="to_logistic">Outgoing message to <c>logistic</c>.</param>
		/// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
		/// <remarks>
		///   <para>The formula for the result is <c>log(sum_(logistic,x) p(logistic,x) factor(logistic,x) / sum_logistic p(logistic) messageTo(logistic))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
		/// </remarks>
		[Skip]
		public static double LogEvidenceRatio(Beta logistic, Gaussian x, Gaussian falseMsg, [Fresh] Beta to_logistic){
			Console.WriteLine("LogEvidenceRatio(Beta logistic, Gaussian x, Gaussian falseMsg, [Fresh] Beta to_logistic)");
			// always zero when using the stabilized message from LogisticAverageConditional
			return 0.0;
			//return LogAverageFactor(logistic, x, falseMsg) - to_logistic.GetLogAverageOf(logistic);
		}

		/// <summary />
		/// <returns />
		/// <remarks>
		///   <para />
		/// </remarks>
		[Skip]
		public static Beta LogisticAverageConditionalInit(){
			Console.WriteLine("LogisticAverageConditionalInit()");
			return Beta.Uniform();
		}



		#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
		






#pragma warning disable 162
		#endif

		/// <summary>
		/// Find a Beta distribution with given integral and mean times a Beta weight function.
		/// </summary>
		/// <param name="mean">The desired value of the mean</param>
		/// <param name="logZ">The desired value of the integral</param>
		/// <param name="a">trueCount-1 of the weight function</param>
		/// <param name="b">falseCount-1 of the weight function</param>
		/// <returns></returns>
		private static Beta BetaFromMeanAndIntegral(double mean, double logZ, double a, double b){
			// The constraints are:
			// 1. int_p to_p(p) p^a (1-p)^b dp = exp(logZ)
			// 2. int_p to_p(p) p p^a (1-p)^b dp = mean*exp(logZ)
			// Let to_p(p) = Beta(p; af, bf)
			// The LHS of (1) is gamma(af+bf)/gamma(af+bf+a+b) gamma(af+a)/gamma(af) gamma(bf+b)/gamma(bf)
			// The LHS of (2) is gamma(af+bf)/gamma(af+bf+a+b+1) gamma(af+a+1)/gamma(af) gamma(bf+b)/gamma(bf)
			// The ratio of (2)/(1) is gamma(af+a+1)/gamma(af+a) gamma(af+bf+a+b)/gamma(af+bf+a+b+1) = (af+a)/(af+bf+a+b) = mean
			// Solving for bf gives bf = (af+a)/mean - (af+a+b).
			// To solve for af, we apply a generalized Newton algorithm to solve equation (1) with bf substituted.
			// af0 is the smallest value of af that ensures (af >= 0, bf >= 0).
			if(mean <= 0)
				throw new ArgumentException("mean <= 0");
			if(mean >= 1)
				throw new ArgumentException("mean >= 1");
			if(double.IsNaN(mean))
				throw new ArgumentException("mean is NaN");
			// bf = (af+bx)*(1-m)/m
			double bx = -(mean * (a + b) - a) / (1 - mean);
			// af0 is the lower bound for af
			// we need both af>0 and bf>0
			double af0 = Math.Max(0, -bx);
			double x = Math.Max(0, bx);
			double af = af0 + 1; // initial guess for af
			double invMean = 1 / mean;
			double bf = (af + a) * invMean - (af + a + b);
			for(int iter = 0; iter < 20; iter++){
				double old_af = af;
				double f = (MMath.GammaLn(af + bf) - MMath.GammaLn(af + bf + a + b)) + (MMath.GammaLn(af + a) - MMath.GammaLn(af)) +
				           (MMath.GammaLn(bf + b) - MMath.GammaLn(bf));
				double g = (MMath.Digamma(af + bf) - MMath.Digamma(af + bf + a + b)) * invMean + (MMath.Digamma(af + a) - MMath.Digamma(af)) +
				           (MMath.Digamma(bf + b) - MMath.Digamma(bf)) * (invMean - 1);
				// fit a fcn of the form: s*log((af-af0)/(af+x)) + c
				// whose deriv is s/(af-af0) - s/(af+x)
				double s = g / (1 / (af - af0) - 1 / (af + x));
				double c = f - s * Math.Log((af - af0) / (af + x));
				bool isIncreasing = (x > -af0);
				if((!isIncreasing && c >= logZ) || (isIncreasing && c <= logZ)){
					// the approximation doesn't fit; use Gauss-Newton instead
					af += (logZ - f) / g;
				} else{
					// now solve s*log((af-af0)/(af+x))+c = logz
					// af-af0 = exp((logz-c)/s) (af+x)
					af = af0 + (x + af0) / MMath.ExpMinus1((c - logZ) / s);
					if(af == af0)
						throw new ArgumentException("logZ is out of range");
				}
				if(double.IsNaN(af))
					throw new ApplicationException("af is nan");
				bf = (af + a) / mean - (af + a + b);
				if(Math.Abs(af - old_af) < 1e-8)
					break;
			}
			if(false){
				// check that integrals are correct
				double f = (MMath.GammaLn(af + bf) - MMath.GammaLn(af + bf + a + b)) + (MMath.GammaLn(af + a) - MMath.GammaLn(af)) +
				           (MMath.GammaLn(bf + b) - MMath.GammaLn(bf));
				if(Math.Abs(f - logZ) > 1e-6)
					throw new ApplicationException("wrong f");
				double f2 = (MMath.GammaLn(af + bf) - MMath.GammaLn(af + bf + a + b + 1)) + (MMath.GammaLn(af + a + 1) - MMath.GammaLn(af)) +
				            (MMath.GammaLn(bf + b) - MMath.GammaLn(bf));
				if(Math.Abs(f2 - (Math.Log(mean) + logZ)) > 1e-6)
					throw new ApplicationException("wrong f2");
			}
			return new Beta(af,bf);
		}

		#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
		






#pragma warning restore 162
		#endif

		/// <summary>EP message to <c>x</c>.</summary>
		/// <param name="logistic">Constant value for <c>logistic</c>.</param>
		/// <returns>The outgoing EP message to the <c>x</c> argument.</returns>
		/// <remarks>
		///   <para>The outgoing message is the factor viewed as a function of <c>x</c> conditioned on the given values.</para>
		/// </remarks>
		public static Gaussian XAverageConditional(double logistic){
			Console.WriteLine("XAverageConditional(double logistic)");
			return Gaussian.PointMass(MMath.Logit(logistic));
		}

		/// <summary>Initialize the buffer <c>falseMsg</c>.</summary>
		/// <returns>Initial value of buffer <c>falseMsg</c>.</returns>
		/// <remarks>
		///   <para />
		/// </remarks>
		[Skip]
		public static Gaussian FalseMsgInit(){
			Console.WriteLine("FalseMsgInit()");
			return new Gaussian();
		}

		/// <summary>Update the buffer <c>falseMsg</c>.</summary>
		/// <param name="logistic">Incoming message from <c>logistic</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <param name="falseMsg">Buffer <c>falseMsg</c>.</param>
		/// <returns>New value of buffer <c>falseMsg</c>.</returns>
		/// <remarks>
		///   <para />
		/// </remarks>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="logistic" /> is not a proper distribution.</exception>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="x" /> is not a proper distribution.</exception>
		public static Gaussian FalseMsg([SkipIfUniform] Beta logistic, 
		                                [Proper] Gaussian x, Gaussian falseMsg){
			if(IsPrintLog){
				Console.WriteLine("FalseMsg([SkipIfUniform] Beta logistic, [Proper] Gaussian x, Gaussian falseMsg)");
			}

			// falseMsg approximates sigma(-x)
			// logistic(sigma(x)) N(x;m,v)
			// = sigma(x)^(a-1) sigma(-x)^(b-1) N(x;m,v) 
			// = e^((a-1)x) sigma(-x)^(a+b-2) N(x;m,v)
			// = sigma(-x)^(a+b-2) N(x;m+(a-1)v,v) exp((a-1)m + (a-1)^2 v/2)
			// = sigma(-x) (prior)
			// where prior = sigma(-x)^(a+b-3) N(x;m+(a-1)v,v)
			double tc1 = logistic.TrueCount - 1;
			double fc1 = logistic.FalseCount - 1;
			double m, v;
			x.GetMeanAndVariance(out m, out v);
			if(tc1 + fc1 == 0){
				falseMsg.SetToUniform();
				return falseMsg;
			} else if(tc1 + fc1 < 0){
				// power EP update, using 1/sigma(-x) as the factor
				Gaussian prior = new Gaussian(m + tc1 * v,v) * (falseMsg ^ (tc1 + fc1 + 1));
				double mprior, vprior;
				prior.GetMeanAndVariance(out mprior, out vprior);
				// posterior moments can be computed exactly
				double w = MMath.Logistic(mprior + 0.5 * vprior);
				Gaussian post = new Gaussian(mprior + w * vprior,vprior * (1 + w * (1 - w) * vprior));
				return prior / post;
			} else{
				// power EP update
				Gaussian prior = new Gaussian(m + tc1 * v,v) * (falseMsg ^ (tc1 + fc1 - 1));
				Gaussian newMsg = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, prior);
				//Console.WriteLine("prior = {0}, falseMsg = {1}, newMsg = {2}", prior, falseMsg, newMsg);
				if(true){
					// adaptive damping scheme
					Gaussian ratio = newMsg / falseMsg;
					if((ratio.MeanTimesPrecision < 0 && prior.MeanTimesPrecision > 0) ||
					   (ratio.MeanTimesPrecision > 0 && prior.MeanTimesPrecision < 0)){
						// if the update would change the sign of the mean, take a fractional step so that the new prior has exactly zero mean
						// newMsg = falseMsg * (ratio^step)
						// newPrior = prior * (ratio^step)^(tc1+fc1-1)
						// 0 = prior.mp + ratio.mp*step*(tc1+fc1-1)
						double step = -prior.MeanTimesPrecision / (ratio.MeanTimesPrecision * (tc1 + fc1 - 1));
						if(step > 0 && step < 1){
							newMsg = falseMsg * (ratio ^ step);
							// check that newPrior has zero mean
							//Gaussian newPrior = prior * ((ratio^step)^(tc1+fc1-1));
							//Console.WriteLine(newPrior);
						}
					}
				}
				return newMsg;
			}
		}

	

		//-- VMP -------------------------------------------------------------------------------------------------

		/// <summary>Evidence message for VMP.</summary>
		/// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <param name="logistic">Incoming message from <c>logistic</c>.</param>
		/// <param name="to_logistic">Previous outgoing message to <c>logistic</c>.</param>
		/// <returns>Zero.</returns>
		/// <remarks>
		///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
		/// </remarks>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="x" /> is not a proper distribution.</exception>
		//[Skip]
		public static double AverageLogFactor([Proper, SkipIfUniform] Gaussian x, Beta logistic, Beta to_logistic){
			Console.WriteLine("AverageLogFactor([Proper, SkipIfUniform] Gaussian x, Beta logistic, Beta to_logistic)");
			double m, v;
			x.GetMeanAndVariance(out m, out v);
			double l1pe = v == 0 ? MMath.Log1PlusExp(m) : MMath.Log1PlusExpGaussian(m, v);
			return (logistic.TrueCount - 1.0) * (m - l1pe) + (logistic.FalseCount - 1.0) * (-l1pe) - logistic.GetLogNormalizer() - to_logistic.GetAverageLog(logistic);
		}

		/// <summary>VMP message to <c>logistic</c>.</summary>
		/// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <returns>The outgoing VMP message to the <c>logistic</c> argument.</returns>
		/// <remarks>
		///   <para>The outgoing message is a distribution matching the moments of <c>logistic</c> as the random arguments are varied. The formula is <c>proj[sum_(x) p(x) factor(logistic,x)]</c>.</para>
		/// </remarks>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="x" /> is not a proper distribution.</exception>
		public static Beta LogisticAverageLogarithm([Proper] Gaussian x){
			Console.WriteLine("LogisticAverageLogarithm([Proper] Gaussian x)");
			double m, v;
			x.GetMeanAndVariance(out m, out v);

			#if true
			// for consistency with XAverageLogarithm
			double eLogOneMinusP = BernoulliFromLogOddsOp.AverageLogFactor(false, x);
			#else
			// E[log (1-sigma(x))] = E[log sigma(-x)] = -E[log(1+exp(x))]
			double eLogOneMinusP = -MMath.Log1PlusExpGaussian(m, v);
			#endif
			// E[log sigma(x)] = -E[log(1+exp(-x))] = -E[log(1+exp(x))-x] = -E[log(1+exp(x))] + E[x]
			double eLogP = eLogOneMinusP + m;
			return Beta.FromMeanLogs(eLogP, eLogOneMinusP);
		}

		/// <summary>VMP message to <c>x</c>.</summary>
		/// <param name="logistic">Incoming message from <c>logistic</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <param name="x">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
		/// <param name="to_X">Previous outgoing message to <c>X</c>.</param>
		/// <returns>The outgoing VMP message to the <c>x</c> argument.</returns>
		/// <remarks>
		///   <para>The outgoing message is the factor viewed as a function of <c>x</c> with <c>logistic</c> integrated out. The formula is <c>sum_logistic p(logistic) factor(logistic,x)</c>.</para>
		/// </remarks>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="logistic" /> is not a proper distribution.</exception>
		/// <exception cref="ImproperMessageException">
		///   <paramref name="x" /> is not a proper distribution.</exception>
		public static Gaussian XAverageLogarithm([SkipIfUniform] Beta logistic, [Proper, SkipIfUniform] Gaussian x, Gaussian to_X){
			Console.WriteLine("XAverageLogarithm([SkipIfUniform] Beta logistic, [Proper, SkipIfUniform] Gaussian x, Gaussian to_X)");
			if(logistic.IsPointMass)
				return XAverageLogarithm(logistic.Point);
			// f(x) = sigma(x)^(a-1) sigma(-x)^(b-1)
			//      = sigma(x)^(a+b-2) exp(-x(b-1))
			// since sigma(-x) = sigma(x) exp(-x)

			double a = logistic.TrueCount;
			double b = logistic.FalseCount;
			double scale = a + b - 2;
			if(scale == 0.0)
				return Gaussian.Uniform();
			double shift = -(b - 1);
			Gaussian toLogOddsPrev = Gaussian.FromNatural((to_X.MeanTimesPrecision - shift) / scale, to_X.Precision / scale);
			Gaussian toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageLogarithm(true, x, toLogOddsPrev);
			return Gaussian.FromNatural(scale * toLogOdds.MeanTimesPrecision + shift, scale * toLogOdds.Precision);
		}

		/// <summary>VMP message to <c>x</c>.</summary>
		/// <param name="logistic">Constant value for <c>logistic</c>.</param>
		/// <returns>The outgoing VMP message to the <c>x</c> argument.</returns>
		/// <remarks>
		///   <para>The outgoing message is the factor viewed as a function of <c>x</c> conditioned on the given values.</para>
		/// </remarks>
		public static Gaussian XAverageLogarithm(double logistic){
			Console.WriteLine("XAverageLogarithm(double logistic)");
			return XAverageConditional(logistic);
		}
	}

}

