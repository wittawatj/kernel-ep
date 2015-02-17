#light

namespace TwoCoinsTutorial

open System
open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Models
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Factors
open MicrosoftResearch.Infer.FSharp

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for two coins tutorial
//-----------------------------------------------------------------------------------

module coins = 
   
    let twoCoinsTestFunc() = 
        Console.WriteLine("\n========================Running Two Coins Tutorial========================\n");
        let firstCoin = Variable.Bernoulli(0.5)
        let secondCoin = Variable.Bernoulli(0.5)
        let bothHeads = firstCoin &&& secondCoin

        // The inference
        let ie = InferenceEngine()

        let bothHeadsPost = ie.Infer<Bernoulli>(bothHeads)
        printfn "Both heads posterior = %A" bothHeadsPost
        bothHeads.ObservedValue <- false

        let firstCoinPost = ie.Infer<Bernoulli>(firstCoin)
        printfn "First coin posterior = %A" firstCoinPost



