using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    static class NetworkEvaluator
    {
        private static double classificationError;

        public static double ClassificationError
        {
            get { return classificationError; }
        }

        private static int errorCode = 0;

        public static double Run(NeuralNetwork Network, DataSet TestSet)
        {
            // TO-DO: transform this into parallelized GPU code
            int nCorrectClassifications = 0;
            float [] trafficSign;
            int label;
            float[] outputClassScores;
            int assignedClass;

            for (int iSign = 0; iSign < TestSet.Size; iSign++ )
            {
                trafficSign = TestSet.TrafficSign(iSign);
                label = TestSet.Label(iSign);

                errorCode = Network.RunForwardOne(trafficSign, out outputClassScores);
                // check error code
                assignedClass = Utils.IndexOfMax(outputClassScores);

                if (assignedClass == label)
                    nCorrectClassifications += 1;
            }

            return (double)nCorrectClassifications / (double)TestSet.Size;
        }


    }
}
