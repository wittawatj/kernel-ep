using System;
using System.Linq;

namespace KernelEP.TestIdea{
	public class TestLinq{

		public static void ArrayFilter1(){
			int[] numbers = {2,1,5,3,8,4};
			var lowNums = 
				from n in numbers 
				where n <= 3 
				select n;
			foreach(var n in lowNums){
				Console.WriteLine(n);
			}

		}
	}
}

