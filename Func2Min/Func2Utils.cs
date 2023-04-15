using MathNet.Numerics;
using MathNet.Numerics.Differentiation;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Func2Min
{
    internal static class Func2Utils
    {
        internal static Vector<double> Grad(Func<Vector<double>, double> f,
                                            Vector<double> point)
        {
            return point.MapIndexed<double>((i, x) => Differentiate.FirstPartialDerivative(p => f(CreateVector.Dense(p)), 
                                            point.AsArray(), i));
        }

        internal static double LamdaS(Func<Vector<double>, double> f, Vector<double> point,
                                      Vector<double> s)
        {
            return -1.0 * (Grad(f, point) * s) / ((s * Hesse(f, point)) * s);
        }

        private static Matrix<double> Hesse(Func<Vector<double>, double> f, Vector<double> point)
        {
            // double[] wrappers
            Func<double[], double> arrf = p => f(CreateVector.Dense(p));
            double[] arrp = point.AsArray();

            // Derivatives
            NumericalDerivative nd = new NumericalDerivative();
            double h11 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 0, 0 }, 2);
            double h12 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 0, 1 }, 2);
            double h21 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 1, 0 }, 2);
            double h22 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 1, 1 }, 2);

            // Returning
            return DenseMatrix.OfArray(new double[,] { { h11, h12 },
                                                       { h21, h22 } });
        }
    }
}
