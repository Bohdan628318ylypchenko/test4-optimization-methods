using Func2Minimizers;
using MathNet.Numerics.LinearAlgebra;

namespace Func2Min
{
    public class FletcherReevesFunc2Minimizer : Func2Minimizer
    {
        private Vector<double> prevGrad = null;

        private Vector<double> prevS = null;

        private bool isFirstSCall = true;

        protected override Vector<double> S(Func<Vector<double>, double> f, Vector<double> point)
        {
            // Current grad
            Vector<double> grad;
            
            // Check for 1st s calculation
            if (isFirstSCall)
            {
                // 1st time call -> grad, s
                isFirstSCall = false;
                prevGrad = Func2Utils.Grad(f, point);
                prevS = -1.0 * prevGrad;
                return prevS;
            }
            else
            {
                // Calc as usual
                grad = Func2Utils.Grad(f, point);
                return -1.0 * grad + prevS * (Math.Pow(grad.L2Norm(), 2) / Math.Pow(prevGrad.L2Norm(), 2));
            }
        }
    }
}
