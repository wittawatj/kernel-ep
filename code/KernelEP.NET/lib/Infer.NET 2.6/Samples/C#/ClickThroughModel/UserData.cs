using System;
using System.Collections.Generic;
using System.Text;

namespace ClickThroughModel
{
	enum clickType
	{
		TT, TF, FT, FF

	};

	enum probType
	{
		NoClick,  ClickNotRel,  ClickRel
	};



	class UserData
	{
		public int[] nClicks = new int[4];

		public double[] probExamine = new double[3];
		public bool[][] clicks;
		public int nIters;
		public int nUsers;

		public override string ToString()
		{
			return String.Format("nTT ={0}, nTF = {1}, nFT = {2}, nFF ={3}", nClicks[(int)clickType.TT], nClicks[(int)clickType.TF], nClicks[(int)clickType.FT], nClicks[(int)clickType.FF]);
		}
	}
}
