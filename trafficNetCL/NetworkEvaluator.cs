using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JaNet
{
    static class NetworkEvaluator
    {
        private static double classificationError;

        public static double ClassificationError
        {
            get { return classificationError; }
        }

        private static int errorCode = 0;

        [Obsolete("Old method")]
        public static double Run(NeuralNetwork Network, DataSet TestSet)
        {
            // TO-DO: transform this into parallelized GPU code
            int nCorrectClassifications = 0;
            float [] trafficSign;
            int label;
            float[] outputClassScores;
            int assignedClass;

            for (int iDataPoint = 0; iDataPoint < TestSet.Size; iDataPoint++ )
            {
                trafficSign = TestSet.GetDataPoint(iDataPoint);
                label = TestSet.GetLabel(iDataPoint);

                errorCode = Network.RunForwardOne(trafficSign, out outputClassScores);
                // check error code

                assignedClass = Utils.IndexOfMax(outputClassScores);

                if (assignedClass == label)
                    nCorrectClassifications += 1;
            }

            return (double)nCorrectClassifications / (double)TestSet.Size;
        }


        public static double ComputeClassificationError(NeuralNetwork network, DataSet dataSet)
        {
            double classificationError = 0;

            for (int i = 0; i < dataSet.Size; i++)
            {
                network.Layers[0].Input.SetHost(dataSet.GetDataPoint(i));

                // Run forward
                for (int l = 0; l < network.Layers.Count; l++)
                {
                    network.Layers[l].FeedForward();
                }

                // Check for correct/wrong classification
                int outputClassMaxScore = Utils.IndexOfMax(network.Layers.Last().Output.GetHost());
                if (outputClassMaxScore != dataSet.GetLabel(i))
                {
                    classificationError += 1;
                }
            }

            return classificationError / dataSet.Size;
        }


    }
}
