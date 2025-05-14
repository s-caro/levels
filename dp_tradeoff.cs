using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.WindowsForms;
using OxyPlot.Axes;

class DPTradeoff
{
    const int R = 5;
    const int K = 6;
    static double step = 1.0 / 100;
    static double startingPoint = 0;
    static int precision = 8;
    static double allowedError = Math.Pow(10, -precision);
    static List<double> s_values = GenerateFloatRange(startingPoint, 1, step);
    
    static Dictionary<Tuple<int, double>, double> T = new Dictionary<Tuple<int, double>, double>();
    static Dictionary<Tuple<int, double, double, int, double>, double> P = new Dictionary<Tuple<int, double, double, int, double>, double>();

    static double TruncateFloat(double value, double step)
    {
        int precision = step.ToString().Split('.')[1].Length;
        return Math.Round(value, precision);
    }

    static List<double> GenerateFloatRange(double start, double end, double step)
    {
        if (step == 0)
        {
            return start == end ? new List<double> { start } : new List<double>();
        }

        if ((start > end && step > 0) || (start < end && step < 0))
        {
            return new List<double>();
        }

        List<double> result = new List<double>();
        double current = start;

        while ((step > 0 && current <= end) || (step < 0 && current >= end))
        {
            result.Add(TruncateFloat(current, step));
            current += step;
        }

        if (TruncateFloat(current - step, step) != TruncateFloat(end, step))
        {
            result.Add(TruncateFloat(end, step));
        }

        return result.Distinct().OrderBy(x => x).ToList();
    }

    static double FindClosestValue(double targetValue)
    {
        if (s_values.Count == 0) return double.NaN;

        if (targetValue <= s_values[0]) return s_values[0];
        if (targetValue >= s_values.Last()) return s_values.Last();

        int left = 0, right = s_values.Count - 1;
        while (left < right - 1)
        {
            int mid = (left + right) / 2;
            if (Math.Abs(s_values[mid] - targetValue) < double.Epsilon)
                return targetValue;
            else if (s_values[mid] < targetValue)
                left = mid;
            else
                right = mid;
        }

        return Math.Abs(s_values[left] - targetValue) <= Math.Abs(s_values[right] - targetValue) 
            ? s_values[left] : s_values[right];
    }

    static List<double> ExtractRangeFromClosest(double start, double end)
    {
        if (s_values.Count == 0) return new List<double>();

        double closestStart = FindClosestValue(start);
        double closestEnd = FindClosestValue(end);

        int startIndex = s_values.IndexOf(closestStart);
        int endIndex = s_values.IndexOf(closestEnd);

        if (startIndex > endIndex) return new List<double>();
        
        if (Math.Abs(end - 0.5) < double.Epsilon && s_values[endIndex] < 0.5)
        {
            var res = s_values.Skip(startIndex).Take(endIndex - startIndex + 1).ToList();
            res.Add(0.5);
            return res;
        }
        
        if (Math.Abs(end - 0.5) < double.Epsilon && endIndex + 1 < s_values.Count && s_values[endIndex + 1] >= 0.5)
        {
            var res = s_values.Skip(startIndex).Take(endIndex - startIndex).ToList();
            res.Add(0.5);
            return res;
        }

        return s_values.Skip(startIndex).Take(endIndex - startIndex + 1).ToList();
    }

    static double H(double x)
    {
        if (x == 0 || x == 1) return 0;
        return -(x * Math.Log(x, 2) + (1 - x) * Math.Log(1 - x, 2));
    }

    static double HInverse(double s, double allowedError)
    {
        double left = 0, right = 0.5;
        while (right - left > allowedError)
        {
            double m1 = left + (right - left) / 3;
            double m2 = right - (right - left) / 3;
            double H_m1 = H(m1);
            double H_m2 = H(m2);

            if (Math.Abs(H_m1 - s) < allowedError) return m1;
            if (Math.Abs(H_m2 - s) < allowedError) return m2;

            if (H_m1 < s) left = m1;
            else right = m2;
        }
        return (left + right) / 2;
    }

    static void PlotResults()
    {
        var plotModel1 = new PlotModel { Title = "Scatter plot of T values for different r (auto scale)" };
        var plotModel2 = new PlotModel { Title = "Scatter plot of T values for different r" };
        var plotModel3 = new PlotModel { Title = $"Comparison: Samples {1/step} - {K} alpha values" };

        // Plot 1 and 2 - T values
        for (int r = 0; r <= R; r++)
        {
            var sVals = T.Keys.Where(k => k.Item1 == r).Select(k => k.Item2).ToList();
            var tVals = sVals.Select(s => T[Tuple.Create(r, s)]).ToList();

            var series1 = new ScatterSeries { MarkerType = MarkerType.Circle, Title = $"r={r}" };
            var series2 = new ScatterSeries { MarkerType = MarkerType.Circle, Title = $"r={r}" };

            for (int i = 0; i < sVals.Count; i++)
            {
                series1.Points.Add(new ScatterPoint(sVals[i], tVals[i]));
                series2.Points.Add(new ScatterPoint(sVals[i], tVals[i]));
            }

            plotModel1.Series.Add(series1);
            plotModel2.Series.Add(series2);
        }

        plotModel1.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "s values" });
        plotModel1.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "T values" });
        plotModel2.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "s values" });
        plotModel2.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "T values", Minimum = 0, Maximum = 1.1 });

        // Plot 3 - Comparison
        var xValues = T.Keys.Where(k => k.Item1 == R).Select(k => Math.Pow(2, k.Item2)).ToList();
        var yValues = T.Keys.Where(k => k.Item1 == R).Select(k => Math.Pow(2, T[k])).ToList();

        var scatterSeries = new ScatterSeries { MarkerType = MarkerType.Circle, Title = "2^(T[7,s]) (scatter)" };
        var lineSeries = new LineSeries { Title = "2^(T[7,s]) (line)" };
        var comparisonSeries = new LineSeries { Title = "Time-Space Complexity (line)" };

        for (int i = 0; i < xValues.Count; i++)
        {
            scatterSeries.Points.Add(new ScatterPoint(xValues[i], yValues[i]));
            lineSeries.Points.Add(new DataPoint(xValues[i], yValues[i]));
        }

        for (double k = 0; k <= 0.5; k += 0.01)
        {
            double timeVal = Math.Pow(1.12717, k) * Math.Pow(1.82653, 1);
            double spaceVal = Math.Pow(0.79703, k) * Math.Pow(1.82653, 1);
            comparisonSeries.Points.Add(new DataPoint(spaceVal, timeVal));
        }

        plotModel3.Series.Add(scatterSeries);
        plotModel3.Series.Add(lineSeries);
        plotModel3.Series.Add(comparisonSeries);
        plotModel3.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Space Complexity" });
        plotModel3.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Time Complexity" });

        // Save plots
        string outputDir = Path.Combine(Directory.GetCurrentDirectory(), $"{K}\\dp\\");
        Directory.CreateDirectory(outputDir);

        var exporter1 = new PngExporter { Width = 1200, Height = 600 };
        exporter1.ExportToFile(plotModel1, Path.Combine(outputDir, $"T_values_s_{step}_k_{K}_R_{R}.png"));
        exporter1.ExportToFile(plotModel2, Path.Combine(outputDir, $"T_values_fixed_s_{step}_k_{K}_R_{R}.png"));
        exporter1.ExportToFile(plotModel3, Path.Combine(outputDir, $"Comparison_s_{step}_k_{K}_R_{R}.png"));

        // Save data
        using (StreamWriter writer = new StreamWriter(Path.Combine(outputDir, $"T_values_k_{K}_R_{R}_{step}.txt")))
        {
            foreach (var key in T.Keys)
            {
                writer.WriteLine($"T[{key.Item1}, {key.Item2}] = {T[key]}");
            }
        }
    }

    static void Main()
    {
        DateTime vStartTime = DateTime.Now;

        // Initialize base cases
        foreach (double s in s_values)
        {
            T[Tuple.Create(0, s)] = 1;
            double hInverse = HInverse(s, allowedError);
            List<double> alpha1Values = ExtractRangeFromClosest(0, FindClosestValue(hInverse));
            
            for (int r = 1; r <= R; r++)
            {
                foreach (double alpha1 in alpha1Values)
                {
                    P[Tuple.Create(r, s, alpha1, 1, alpha1)] = 0;
                }
            }
        }

        // Main dynamic programming loops
        for (int r = 1; r <= R; r++)
        {
            for (int i = 2; i <= K + 1; i++)
            {
                if (r > 1 || (r == 1 && i == 7))
                {
                    DateTime startTime = DateTime.Now;
                    
                    foreach (double s in s_values)
                    {
                        Console.WriteLine($"start s-r-k: {s}-{r}-{i} at time {DateTime.Now:dd/MMM, HH:mm:ss}");
                        double hInverse = HInverse(s, allowedError);
                        List<double> alpha1Values = ExtractRangeFromClosest(0, FindClosestValue(hInverse));
                        
                        foreach (double alpha1 in alpha1Values)
                        {
                            if (i == 2)
                            {
                                List<double> alpha2Values = ExtractRangeFromClosest(FindClosestValue(alpha1), 0.5);
                                foreach (double alpha2 in alpha2Values)
                                {
                                    if (alpha2 == 0 || Math.Abs(alpha2 - alpha1) < double.Epsilon)
                                    {
                                        P[Tuple.Create(r, s, alpha1, 2, alpha2)] = 0;
                                    }
                                    else
                                    {
                                        P[Tuple.Create(r, s, alpha1, 2, alpha2)] = 
                                            0.5 * alpha2 * H(alpha1 / alpha2) + 
                                            (alpha2 - alpha1) * T[Tuple.Create(r - 1, Math.Min(FindClosestValue(s / (alpha2 - alpha1)), 1))];
                                    }
                                }
                            }
                            else if (i >= 3)
                            {
                                List<double> alphaIValues = ExtractRangeFromClosest(FindClosestValue(alpha1), 0.5);
                                foreach (double alphaI in alphaIValues)
                                {
                                    List<double> alphaI1Values = ExtractRangeFromClosest(FindClosestValue(alpha1), FindClosestValue(alphaI));
                                    P[Tuple.Create(r, s, alpha1, i, alphaI)] = 1;
                                    double P_temp = 1;
                                    
                                    foreach (double alphaI1 in alphaI1Values)
                                    {
                                        if (alphaI == 0)
                                        {
                                            P_temp = 0;
                                        }
                                        else if (Math.Abs(alphaI - alphaI1) < double.Epsilon)
                                        {
                                            P_temp = P[Tuple.Create(r, s, alpha1, i - 1, alphaI1)];
                                        }
                                        else
                                        {
                                            P_temp = 
                                                0.5 * alphaI * H(alphaI1 / alphaI) + 
                                                Math.Max(P[Tuple.Create(r, s, alpha1, i - 1, alphaI1)],
                                                    (alphaI - alphaI1) * 
                                                    T[Tuple.Create(r - 1, Math.Min(FindClosestValue(s / (alphaI - alphaI1)), 1))]);
                                        }
                                        
                                        P[Tuple.Create(r, s, alpha1, i, alphaI)] = 
                                            Math.Min(P[Tuple.Create(r, s, alpha1, i, alphaI)], P_temp);
                                    }
                                }
                            }
                        }
                    }

                    // Save partial results
                    string outputDir = Path.Combine(Directory.GetCurrentDirectory(), $"{K}\\dp\\");
                    Directory.CreateDirectory(outputDir);
                    
                    using (StreamWriter writer = new StreamWriter(Path.Combine(outputDir, $"P_values_k_{K}_R_{R}_{step}_partial_{r}_{i}.txt")))
                    {
                        foreach (var key in P.Keys)
                        {
                            writer.WriteLine($"P[{key.Item1},{key.Item2},{key.Item3},{key.Item4},{key.Item5}] = {P[key]}");
                        }
                    }

                    TimeSpan elapsedTime = DateTime.Now - startTime;
                    Console.WriteLine($"now: {DateTime.Now:dd/MMM, HH:mm:ss} - step: {step} - r: {r} - i: {i} - in time: {elapsedTime:hh\\:mm\\:ss}");
                }
            }

            // Update T values
            foreach (double s in s_values)
            {
                double hInverse = HInverse(s, allowedError);
                List<double> alpha1Values = ExtractRangeFromClosest(0, FindClosestValue(hInverse));
                T[Tuple.Create(r, s)] = 1;
                double T_temp = 1;
                
                foreach (double alpha1 in alpha1Values)
                {
                    T_temp = Math.Max(H(alpha1), 0.5 + P[Tuple.Create(r, s, alpha1, K + 1, 0.5)]);
                    T[Tuple.Create(r, s)] = Math.Min(T[Tuple.Create(r, s)], T_temp);
                }
            }

            // Save partial T results
            string tOutputDir = Path.Combine(Directory.GetCurrentDirectory(), $"{K}\\dp\\");
            Directory.CreateDirectory(tOutputDir);
            
            using (StreamWriter writer = new StreamWriter(Path.Combine(tOutputDir, $"T_values_k_{K}_R_{R}_{step}_partial_{r}.txt")))
            {
                foreach (var key in T.Keys)
                {
                    writer.WriteLine($"T[{key.Item1}, {key.Item2}] = {T[key]}");
                }
            }
        }

        TimeSpan vElapsedTime = DateTime.Now - vStartTime;
        Console.WriteLine($"now: {DateTime.Now:dd/MMM, HH:mm:ss} - in time: {vElapsedTime:hh\\:mm\\:ss}");

        PlotResults();
    }
}