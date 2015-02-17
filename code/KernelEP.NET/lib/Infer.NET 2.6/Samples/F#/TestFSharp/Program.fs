#light
open System
open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Models
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Factors
open MicrosoftResearch.Infer.FSharp

open TwoCoinsTutorial
open TruncatedGaussianTutorial
open GaussianRangesTutorial
open ClinicalTrialTutorial
open BayesPointTutorial
open MixtureGaussiansTutorial

//main Smoke Test .............................................

let _ = TwoCoinsTutorial.coins.twoCoinsTestFunc()
let _ = TruncatedGaussianTutorial.truncated.truncatedTestFunc() 
let _ = GaussianRangesTutorial.ranges.rangesTestFunc()
let _ = ClinicalTrialTutorial.clinical.clinicalTestFunc()
let _ = BayesPointTutorial.bayes.bayesTestFunc()
let _ = MixtureGaussiansTutorial.mixture.mixtureTestFunc()


Console.ReadLine() |> ignore
