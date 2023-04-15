using Func2Min;
using MathNet.Numerics.LinearAlgebra;

namespace FletcherReevesDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Creating minimizer
            FletcherReevesFunc2Minimizer frm = new ();

            // Function and start point
            Func<Vector<double>, double> f =
            v => 3.0 * Math.Pow(v[0] - 14.0, 2) + v[0] * v[1] + 7.0 * Math.Pow(v[1], 2);
            Vector<double> startPoint = CreateVector.Dense(new double[] { 21.8, 21.8 });

            // Minimizing
            var result = frm.MinimizeFunc(f, startPoint, 1e-5);

            // Printing result
            foreach (var r in result)
            {
                Console.WriteLine(r.At(0).ToString() + " " + r.At(1).ToString());
            }
        }
    }
}