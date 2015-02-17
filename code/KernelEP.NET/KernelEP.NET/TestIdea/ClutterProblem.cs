using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;

namespace KernelEP.TestIdea{

	// This class was written by Nicolas
   class ClutterProblem{
       static void Run(){
           //likelihood
			double dataVar = 1;                 //variance of the "data component" of the mixture
           double clutterMean = 0;             //mean of clutter component
           double clutterVar = 10;             //variance of clutter component
           double w = 0.5;                     //probability of choosing the data component

           //prior
           double priorVar = 100;              //variance of prior over data component mean
           double priorMean = 0;               //mean of prior
           

           //true mean of data component for generating training data
           double trueTheta = 2;

           //number of observed data points
           int nData = 10;

           /**********************************************************************************
            *
            * construct the model -- we're not generating any data here, just defining the
            * model
            *
            **********************************************************************************/
           
           //mixture weights
           double[] mixturePrior = new double[] { w, 1-w };

           //prior over mean
           Variable<double> theta = Variable.GaussianFromMeanAndVariance(priorMean,priorVar);
           
           //model will have n observed data points, obseved data is continuous scalar
           Range n = new Range(nData);
           VariableArray<double> data = Variable.Array<double>(n);

           //define how each of these observed data points are being generated (this is the actual model):
           VariableArray<int> z = Variable.Array<int>(n);      //mixture component indicator for each data point (will be unobserved)
           
           using (Variable.ForEach(n))                         //for each datapoint
           {
               z[n] = Variable.Discrete(mixturePrior );        //choose mixture component by drawing z from discrete distribution with 2 values

               
               using (Variable.Case(z[n], 0))                  //first mixture component
               {
                   //draw data from data component: Gaussian with mean theta (and variance dataVar)
                   data[n] = Variable.GaussianFromMeanAndVariance(theta, dataVar);
               }


               using (Variable.Case(z[n], 1))                  //second mixture component
               {
                   //draw data from clutter component
                   data[n] = Variable.GaussianFromMeanAndVariance(clutterMean, clutterVar);
               }
           }
 
           /****************************************************************************
            *
            * create data by sampling from the model and attach to the variables of the
            * model defined above
            *
            * alternatively read data from matlab file
            *
            ****************************************************************************/

           double[] observedData = new double[nData];
           int[] trueZ = new int[nData];

           if (true)                                           //set to false to read from matlab file
           {
               //sample observed data
               
               Rand.Restart(104);                              //set seed of random number generator to fixed value

               //set up distributions to sample from
               Discrete kDistr = new Discrete(mixturePrior);
               Gaussian clutterDistr = new Gaussian(clutterMean, clutterVar);
               Gaussian dataDistr = new Gaussian(trueTheta, dataVar);

               //generate data
               for (int jj = 0; jj < nData; jj++){
                   trueZ[jj] = kDistr.Sample();

                   if (trueZ[jj] == 0){
                       observedData[jj] = dataDistr.Sample();
                   }else{
                       observedData[jj] = clutterDistr.Sample();
                   }
               }

           }
           else
           {

               // read data from Matlab file that contains an (nx1) matrix variable named "observedData"
               // (note that the size of the matrix must fit the specified number of observed data
               // points from above
               Dictionary<String, Object> fullDataFile = MatlabReader.Read("..\\..\\testData.mat");
               ((Matrix)fullDataFile["observedData"]).CopyTo(observedData, 0);

           }

           //attach the observed data to model variables defined above
           data.ObservedValue = observedData;


           /****************************************************************************
            *
            * run inference
            *
            ****************************************************************************/

           // determine whether we allow improper messages (negative precision)
           // set to false to allow improper messages
           GateEnterOp<double>.ForceProper = true;

           // run inference with Expectation Propagation and show the posterior over theta
           InferenceEngine ie = new InferenceEngine(new ExpectationPropagation());
           Console.WriteLine("Posterior over theta" + ie.Infer(theta));

           // also show the posterior over z for each data point
           DistributionArray<Discrete> zInferred = (DistributionArray<Discrete>) ie.Infer(z);
           Console.WriteLine("\nPosterior over z (true z;  observed value):");
           for (int jj = 0; jj < nData; jj++)
               Console.WriteLine(string.Format("{0:0.00}   ( {1} ; {2:0.00})", zInferred[jj].GetProbs()[0], trueZ[jj], observedData[jj]));

           Console.WriteLine("\n");

       }
   }
}