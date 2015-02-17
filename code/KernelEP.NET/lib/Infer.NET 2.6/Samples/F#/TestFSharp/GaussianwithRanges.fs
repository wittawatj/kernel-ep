#light

namespace GaussianRangesTutorial

open System
open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Models
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Factors
open MicrosoftResearch.Infer.Maths
open MicrosoftResearch.Infer.FSharp

//----------------------------------------------------------------------------------
// Infer.NET: F# script for learning a Gaussian from data
//----------------------------------------------------------------------------------

// The model
module ranges = 
    let rangesTestFunc() = 
        Console.WriteLine("\n======================Running Gaussian Ranges Tutorial=======================\n");
        // The model
        let len = Variable.New<int>()
        let dataRange = Range(len)
        let mean = Variable.GaussianFromMeanAndVariance(0.0, 100.0)
        let precision = Variable.GammaFromShapeAndScale(1.0, 1.0)
        let x = Variable.AssignVariableArray 
                    (Variable.Array<float>(dataRange))  
                     dataRange (fun d -> Variable.GaussianFromMeanAndPrecision(mean, precision))

        // The data
        let data = Array.init 100 (fun _ -> Rand.Normal(0.0, 1.0))

        // Binding the data
        len.ObservedValue <- data.Length
        x.ObservedValue <- data

        // The inference
        let ie = InferenceEngine(VariationalMessagePassing())
        printfn "mean = %A" (ie.Infer<Gaussian>(mean))
        printfn "prec = %A" (ie.Infer<Gamma>(precision))
