// (C) Copyright 2011 Microsoft Research Cambridge
using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using RunCyclingSamples;

namespace InferNET101
{
	public static class InferNET101
	{
		static void Main()
		{
			Console.WriteLine("\n************");
			Console.WriteLine("CyclingTime1");
			Console.WriteLine("************\n");
			RunCyclingSamples.RunCyclingSamples.RunCyclingTime1();
			Console.Write("\nPress the spacebar to continue.");
			Console.ReadKey();

			Console.WriteLine("\n************");
			Console.WriteLine("CyclingTime2");
			Console.WriteLine("************\n");
			RunCyclingSamples.RunCyclingSamples.RunCyclingTime2();
			Console.Write("\nPress the spacebar to continue.");
			Console.ReadKey();

			Console.WriteLine("\n************");
			Console.WriteLine("CyclingTime3");
			Console.WriteLine("************\n");
			RunCyclingSamples.RunCyclingSamples.RunCyclingTime3();
			Console.Write("\nPress the spacebar to continue.");
			Console.ReadKey();

			Console.WriteLine("\n************");
			Console.WriteLine("CyclingTime4");
			Console.WriteLine("************\n");
			RunCyclingSamples.RunCyclingSamples.RunCyclingTime4();
			Console.Write("\nPress the spacebar to continue.");
			Console.ReadKey();

			Console.WriteLine("\n************");
			Console.WriteLine("CyclingTime5");
			Console.WriteLine("************\n");
			RunCyclingSamples.RunCyclingSamples.RunCyclingTime5();
			Console.Write("\nPress the spacebar to continue.");
			Console.ReadKey();
		}
	}
}
