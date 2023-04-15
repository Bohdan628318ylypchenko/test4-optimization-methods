using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Playground
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Matrix<double> I = CreateMatrix.DenseIdentity<double>(2);
            Console.WriteLine(I);
            Matrix<double> Z = CreateMatrix.Dense<double>(2, 2);
            Console.WriteLine(Z);
        }
    }
}