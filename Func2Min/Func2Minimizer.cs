using Func2Min;
using MathNet.Numerics.LinearAlgebra;

namespace Func2Minimizers
{
    public abstract class Func2Minimizer
    {
        public Vector<double>[] MinimizeFunc(Func<Vector<double>, double> f,
                                             Vector<double> startPoint,
                                             double e)
        {
            // List to store path
            LinkedList<Vector<double>> result = new LinkedList<Vector<double>>();
            result.AddLast(startPoint);

            // Minimizing
            Vector<double> currentPoint = startPoint;
            while(!StopCriteria(f, currentPoint, e))
            {
                currentPoint = NextPoint(f, currentPoint);
                result.AddLast(currentPoint);
            }
            
            // Returning
            return result.ToArray();
        }           

        protected Vector<double> NextPoint(Func<Vector<double>, double> f, Vector<double> currentPoint)
        {
            // Calculating s by calling child-specific implementation
            Vector<double> s = S(f, currentPoint);

            // Calculating lambda
            double l = Func2Utils.LamdaS(f, currentPoint, s);

            // Returning
            return currentPoint + l * s;
        }

        protected abstract Vector<double> S(Func<Vector<double>, double> f, Vector<double> point);

        private bool StopCriteria(Func<Vector<double>, double> f,
                                  Vector<double> point,
                                  double e)
        {
            // Gradient module less than e
            return Func2Utils.Grad(f, point).L2Norm() <= e;
        }

    }
}