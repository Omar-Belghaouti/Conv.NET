using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{

    class NetworkEvaluator
    {

        #region Fields

        #endregion

        #region Constructor

        // The constructor does nothing at the moment
        public NetworkEvaluator()
        {

        }
        #endregion

        public void ComputeLossError(NeuralNetwork network, DataSet dataSet, out double loss, out double error)
        {
            // pass network as argument if it doesn't work!

            loss = 0.0;
            error = 0.0;

            int miniBatchSize = network.Layers[0].OutputNeurons.MiniBatchSize;
            
            Sequence indicesSequence = new Sequence(dataSet.Size);

            
            // Run over mini-batches
            for (int iStartMiniBatch = 0; iStartMiniBatch < dataSet.Size; iStartMiniBatch += miniBatchSize)  
            {
                // Feed a mini-batch to the network
                int[] miniBatch = indicesSequence.GetMiniBatchIndices(iStartMiniBatch, miniBatchSize);
                network.InputLayer.FeedData(dataSet, miniBatch);

                // Run network forward
                network.ForwardPass();

                // Find maximum output score (i.e. assigned class) of each mini batch item
                for (int m = 0; m < miniBatchSize; m++)
                {
                    double[] outputScores = network.Layers.Last().OutputClassScores[m];

                    /////////////// DEBUGGING (visualize true label vs network output)
                    /*
                    Console.WriteLine("\n\n----------- Mini batch {0} ---------", iMiniBatch);
                    Console.WriteLine("\n\tData point {0}:", miniBatchItems[iMiniBatchItem]);
                    float[] trueLabelArray = dataSet.GetLabelArray(miniBatchItems[iMiniBatchItem]);
                    Console.WriteLine("\nTrue label:");
                    for (int iClass = 0; iClass < dataSet.NumberOfClasses; iClass++)
                    {
                        Console.Write(trueLabelArray[iClass].ToString("0.##") + " ");
                    }
                    Console.WriteLine();

                    Console.WriteLine("Network output:");
                    for (int iClass = 0; iClass < dataSet.NumberOfClasses; iClass++)
                    {
                        Console.Write(outputScores[iClass].ToString("0.##") + " ");
                    }
                    Console.WriteLine();
                    Console.ReadKey();
                    */
                    ///////////////////

                    int assignedLabel = Utils.IndexOfMax(outputScores);
                    int trueLabel = dataSet.Labels[miniBatch[m]];

                    // Cumulate loss and error
                    loss -= Math.Log(outputScores[trueLabel]);
                    error += (assignedLabel == trueLabel) ? 0 : 1;

                } // end loop within a mini-batch

            } // end loop over mini-batches
             
            error /= dataSet.Size;
            loss /= dataSet.Size;
        }
    }
}
