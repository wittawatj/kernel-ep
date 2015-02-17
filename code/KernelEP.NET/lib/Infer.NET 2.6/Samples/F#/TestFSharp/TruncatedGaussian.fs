#light

namespace TruncatedGaussianTutorial

open System
open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Models
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Factors
open MicrosoftResearch.Infer.Maths
open MicrosoftResearch.Infer.FSharp

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for a truncated Gaussian with a variable threshold
//-----------------------------------------------------------------------------------

// The Model
module truncated =
    let truncatedTestFunc() = 
        Console.WriteLine("\n====================Running Truncated Gaussian Tutorial======================\n");
        let threshold = (Variable.New<float>()).Named("threshold")
        let x = Variable.GaussianFromMeanAndVariance(0.0,1.0).Named("x")
        do Variable.ConstrainTrue( (x >> threshold))

        // The inference, looping over different thresholds
        let ie = InferenceEngine()
        ie.ShowProgress <- false
        threshold.ObservedValue <- -0.1

        for i = 0 to 10 do
          threshold.ObservedValue <- threshold.ObservedValue + 0.1  
          printfn "Dist over x given thresh of %A = %A" threshold.ObservedValue (ie.Infer<Gaussian>(x))
        
