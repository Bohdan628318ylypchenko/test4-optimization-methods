using Func2Minimizers;
using MathNet.Numerics.LinearAlgebra;

namespace Func2Min
{
    public class FletcherReevesFunc2Minimizer : Func2Minimizer
    {
        private Vector<double> _prevGrad = null;

        private Vector<double> _prevS = null;

        private bool _isFirstSCall = true;

        protected override Vector<double> S(Func<Vector<double>, double> f, Vector<double> point)
        {
            // Current grad
            Vector<double> currentGrad;
            Vector<double> currentS;
            
            // Check for 1st s calculation
            if (_isFirstSCall)
            {
                // 1st time call -> grad, s
                _isFirstSCall = false;
                _prevGrad = Func2Utils.Grad(f, point);
                _prevS = -1.0 * _prevGrad;
                return _prevS;
            }
            else
            {
                // Calc as usual
                currentGrad = Func2Utils.Grad(f, point);
                currentS = -1.0 * currentGrad + _prevS * (Math.Pow(currentGrad.L2Norm(), 2) / Math.Pow(_prevGrad.L2Norm(), 2));
                _prevGrad = currentGrad;
                _prevS = currentS;
                return currentS;
            }
        }
    }
}
