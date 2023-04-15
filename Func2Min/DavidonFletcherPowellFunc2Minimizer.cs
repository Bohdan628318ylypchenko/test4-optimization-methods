using Func2Minimizers;
using MathNet.Numerics.LinearAlgebra;

namespace Func2Min
{
    public class DavidonFletcherPowellFunc2Minimizer : Func2Minimizer
    {
        private Vector<double> _prevPoint;

        private Vector<double> _prevGrad;

        private Matrix<double> _prevA;

        private bool _isFirstCall = true;

        protected override Vector<double> S(Func<Vector<double>, double> f, Vector<double> point)
        {
            // Current grad
            Vector<double> currentGrad;

            // Check for 1st time
            if (_isFirstCall)
            {
                // Is 1st time -> init params, return start s
                _isFirstCall = false;
                _prevPoint = point;
                _prevGrad = Func2Utils.Grad(f, _prevPoint);
                _prevA = CreateMatrix.DenseIdentity<double>(2);
                return -1.0 * _prevA * _prevGrad;
            }
            else
            {
                // Calc as usual
                currentGrad = Func2Utils.Grad(f, point);
                Matrix<double> currentA = A(point, currentGrad);
                Console.WriteLine(currentA);
                _prevPoint = point;
                _prevGrad = currentGrad;
                _prevA = currentA;
                return -1.0 * currentA * currentGrad;
            }
        }

        private Matrix<double> A(Vector<double> point, Vector<double> grad)
        {
            // Delta x
            var dx = point - _prevPoint;

            // Delta x as column matrix
            var dxm = CreateMatrix.Dense<double>(2, 2);
            dxm.SetColumn(0, dx);

            // Delta x as row matrix
            var dxmt = CreateMatrix.Dense<double>(2, 2);
            dxmt.SetRow(0, dx);

            // Delta grad
            var dg = grad - _prevGrad;

            // Delta grad as column matrix
            var dgm = CreateMatrix.Dense<double>(2, 2);
            dgm.SetColumn(0, dg);

            // Delta grad as row matrix
            var dgmt = CreateMatrix.Dense<double>(2, 2);
            dgmt.SetRow(0, dg);

            // Calculating A
            return _prevA 
                   + ((dxm * dxmt) / (dxmt * dgm)[0, 0]) 
                   - ((_prevA * dgm * dgmt * _prevA) / (dgmt * _prevA * dgm)[0, 0]);
        }
    }
}
