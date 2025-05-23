﻿// dp_tradeoff_parallel.cs
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.ImageSharp; // To export plots as images

class Program
{
	static readonly int R = 5;
	static readonly int K = 5;
	static readonly double Step = 1.0 / 1000;
	static readonly double StartingPoint = 0;
	static readonly double AllowedError = Math.Pow(10, -8);
	static List<double> s_values = GenerateFloatRange(StartingPoint, 1, Step);

	static ConcurrentDictionary<(int, double), double> T = new();
	static ConcurrentDictionary<(int, double), double> T_each = new();
	static ConcurrentDictionary<(int, double, double, int, double), double> P = new();

	static void Main()
	{
		Console.WriteLine($"Detected cores: {Environment.ProcessorCount}");
		var vStartTime = DateTime.Now;

		foreach (var s in s_values)
		{
			T[(0, s)] = 1;
			double h_inv = HInverse(s, AllowedError);
			foreach (var alpha_1 in ExtractRangeFromClosest(0, FindClosestValue(h_inv)))
			{
				for (int r = 1; r <= R; r++)
					P[(r, s, alpha_1, 1, alpha_1)] = 0;
			}
		}

		Console.WriteLine($"Start - {DateTime.Now:dd/MMM, HH:mm:ss}");

		for (int r = 1; r <= R; r++)
		{
			for (int i = 2; i <= K + 1; i++)
			{
				//int maxParallelism = 110;
				//using (var semaphore = new SemaphoreSlim(maxParallelism))
				//{
				//	var tasks = s_values.Select(async s =>
				//	{
				//		await semaphore.WaitAsync();
				//		try
				//		{
				//			await Task.Run(() => ProcessS(s, r, i));
				//		}
				//		finally
				//		{
				//			semaphore.Release();
				//		}
				//	}).ToArray();

				//	Task.WaitAll(tasks); // Wait for all the throttled tasks to complete
				//}
				var tasks = s_values.Select(s => Task.Run(() => ProcessS(s, r, i))).ToArray();
				Task.WaitAll(tasks);

				Console.WriteLine($"step: {Step} - r: {r} - i: {i} done - {DateTime.Now:dd/MMM, HH:mm:ss}");
				Console.WriteLine($"Start writing dictionary - {DateTime.Now:dd/MMM, HH:mm:ss}");
				foreach (var s in s_values)
				{
					double h_inv = HInverse(s, AllowedError);
					var alpha_1_values = ExtractRangeFromClosest(0, FindClosestValue(h_inv));

					T_each[(r, s)] = 1;
					foreach (var alpha_1 in alpha_1_values)
					{
						double val = Math.Max(H(alpha_1), 0.5 + P[(r, s, alpha_1, i, 0.5)]);
						T_each[(r, s)] = Math.Min(T_each[(r, s)], val);
					}
				}
				// Save partial T results
				string tOutputDir_each = Path.Combine(Directory.GetCurrentDirectory(), $"{i-1}\\dp\\");
				Directory.CreateDirectory(tOutputDir_each);

				using (StreamWriter writer = new StreamWriter(Path.Combine(tOutputDir_each, $"T_values_k_{i-1}_R_{R}_{Step}_partial_{r}_par_with_best_T_each.txt")))
				{

					foreach (var key in T_each.Keys)
					{
						writer.WriteLine($"T[{key.Item1}, {key.Item2}] = {T_each[key]}");
					}
				}
				Console.WriteLine($"End writing dictionary - {DateTime.Now:dd/MMM, HH:mm:ss}");
			}

			foreach (var s in s_values)
			{
				double h_inv = HInverse(s, AllowedError);
				var alpha_1_values = ExtractRangeFromClosest(0, FindClosestValue(h_inv));

				T[(r, s)] = 1;
				foreach (var alpha_1 in alpha_1_values)
				{
					double val = Math.Max(H(alpha_1), 0.5 + P[(r, s, alpha_1, K + 1, 0.5)]);
					T[(r, s)] = Math.Min(T[(r, s)], val);
				}
			}
			// Save partial T results
			string tOutputDir = Path.Combine(Directory.GetCurrentDirectory(), $"{K}\\dp\\");
			Directory.CreateDirectory(tOutputDir);

			using (StreamWriter writer = new StreamWriter(Path.Combine(tOutputDir, $"T_values_k_{K}_R_{R}_{Step}_partial_{r}_par_with_best.txt")))
			{
				
				foreach (var key in T.Keys)
				{
					writer.WriteLine($"T[{key.Item1}, {key.Item2}] = {T[key]}");
				}
			}
		}

		Console.WriteLine($"Finished in {(DateTime.Now - vStartTime).TotalSeconds:F2} seconds.");
	}

	static void ProcessS(double s, int r, int i)
	{
		double h_inv = HInverse(s, AllowedError);
		var alpha_1_values = ExtractRangeFromClosest(0, FindClosestValue(h_inv));

		foreach (var alpha_1 in alpha_1_values)
		{
			if (i == 2)
			{
				if (s >= 0.902)
				{
					P[(r, s, alpha_1, 2, 0.5)] = 0.90144039158 - 0.5;
				}
				else
				{
					var alpha_2_values = ExtractRangeFromClosest(FindClosestValue(alpha_1), 0.5);
					foreach (var alpha_2 in alpha_2_values)
					{
						double val = (alpha_2 == 0 || alpha_2 == alpha_1) ? 0 :
							0.5 * alpha_2 * H(alpha_1 / alpha_2) + (alpha_2 - alpha_1) * T[(r - 1, Math.Min(FindClosestValue(s / (alpha_2 - alpha_1)), 1))];
						P[(r, s, alpha_1, 2, alpha_2)] = val;
					}
				}
			}
			else
			{
				if (i == 3 && s >= 0.869)
				{
					P[(r, s, alpha_1, 3, 0.5)] = 0.86838650181 - 0.5;
				}
				else if (i == 4 && s >= 0.863)
				{
					P[(r, s, alpha_1, 4, 0.5)] = 0.86250296836 - 0.5;
					
				}
				else if (i == 5 && s >= 0.862)
				{
					P[(r, s, alpha_1, 5, 0.5)] = 0.86161399838 - 0.5;
				}
				else if (i == 6 && s >= 0.862)
				{
					P[(r, s, alpha_1, 6, 0.5)] = 0.8614948983 - 0.5;
				}
				else
				{
					var alpha_i_values = ExtractRangeFromClosest(FindClosestValue(alpha_1), 0.5);
					foreach (var alpha_i in alpha_i_values)
					{
						var alpha_i_1_values = ExtractRangeFromClosest(FindClosestValue(alpha_1), FindClosestValue(alpha_i));
						double minP = 1;
						foreach (var alpha_i_1 in alpha_i_1_values)
						{
							double temp = (alpha_i == 0) ? 0 :
								(alpha_i == alpha_i_1) ? P[(r, s, alpha_1, i - 1, alpha_i_1)] :
								0.5 * alpha_i * H(alpha_i_1 / alpha_i) + Math.Max(P[(r, s, alpha_1, i - 1, alpha_i_1)],
								(alpha_i - alpha_i_1) * T[(r - 1, Math.Min(FindClosestValue(s / (alpha_i - alpha_i_1)), 1))]);

							minP = Math.Min(minP, temp);
						}
						P[(r, s, alpha_1, i, alpha_i)] = minP;
					}
				}
			}
		}
	}
	

	static double H(double x) => (x == 0 || x == 1) ? 0 : -x * Math.Log(x, 2) - (1 - x) * Math.Log(1 - x, 2);

	static double HInverse(double s, double error)
	{
		double left = 0, right = 0.5;
		while (right - left > error)
		{
			double m1 = left + (right - left) / 3;
			double m2 = right - (right - left) / 3;
			double Hm1 = H(m1), Hm2 = H(m2);
			if (Math.Abs(Hm1 - s) < error) return m1;
			if (Math.Abs(Hm2 - s) < error) return m2;
			if (Hm1 < s) left = m1; else right = m2;
		}
		return (left + right) / 2;
	}

	static List<double> GenerateFloatRange(double start, double end, double step)
	{
		var range = new List<double>();
		double val = start;
		while (val <= end)
		{
			range.Add(Math.Round(val, 8));
			val += step;
		}
		if (!range.Contains(end)) range.Add(end);
		return range.Distinct().OrderBy(x => x).ToList();
	}

	static double FindClosestValue(double target)
	{
		return s_values.OrderBy(x => Math.Abs(x - target)).First();
	}

	static List<double> ExtractRangeFromClosest(double start, double end)
	{
		double cs = FindClosestValue(start);
		double ce = FindClosestValue(end);
		int si = s_values.IndexOf(cs);
		int ei = s_values.IndexOf(ce);
		var list = s_values.GetRange(Math.Min(si, ei), Math.Abs(ei - si) + 1);
		if (end == 0.5 && !list.Contains(0.5)) list.Add(0.5);
		return list;
	}

}
