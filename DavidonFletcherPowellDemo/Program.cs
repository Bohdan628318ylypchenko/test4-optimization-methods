using Func2Min;
using MathNet.Numerics.LinearAlgebra;

namespace DavidonFletcherPowellDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Creating minimizer
            DavidonFletcherPowellFunc2Minimizer dfpm = new();

            // Function and start point
            Func<Vector<double>, double> f =
            v => 3.0 * Math.Pow(v[0] - 14.0, 2) + v[0] * v[1] + 7.0 * Math.Pow(v[1], 2);
            Vector<double> startPoint = CreateVector.Dense(new double[] { 21.8, 21.8 });

            // Minimizing
            var result = dfpm.MinimizeFunc(f, startPoint, 1e-9);

            // Printing result
            foreach (var r in result)
            {
                Console.WriteLine(r.At(0).ToString() + " " + r.At(1).ToString());
            }

            // Printing H-1
            Matrix<double> H1 = Func2Utils.Hesse(f, result.Last()).Inverse();
            Console.WriteLine(H1);
        }
    }
}