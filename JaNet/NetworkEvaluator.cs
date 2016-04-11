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
        // CLEAN

        #region Fields
#if OPENCL_ENABLED

        private int inputBufferBytesSize;
        private IntPtr[] classificationGlobalWorkSize;
        private IntPtr[] classificationLocalWorkSize;

        private Mem assignedClassBuffer;

#endif
        #endregion

        #region Constructor

        // The constructor does nothing at the moment
        public NetworkEvaluator()
        {
            /*
#if OPENCL_ENABLED
            this.inputBufferBytesSize = sizeof(float) * dataDimension;
            this.classificationGlobalWorkSize = new IntPtr[] { (IntPtr)nClasses };
            this.classificationLocalWorkSize = new IntPtr[] { (IntPtr)nClasses };

            this.assignedClassBuffer = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)sizeof(int),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NetworkEvaluator.SetupCLObjects: Cl.CreateBuffer assignedClassBuffer");
#endif
            */
        }
        #endregion

        public void ComputeLossError(NeuralNetwork network, DataSet dataSet, out double loss, out double error)
        {
            // pass network as argument if it doesn't work!

            loss = 0.0;
            error = 0.0;

            //TODO: do this using OpenCL

            float[] outputScores = new float[dataSet.NumberOfClasses];
            int assignedLabel;
            int trueLabel;
            int outputBufferBytesSize = dataSet.NumberOfClasses * sizeof(float);
            int[] miniBatchItems = new int[network.MiniBatchSize];
            // loop through all data points in dataSet (ordered mini-batches)

            int iMiniBatch = 0;

            // Run over mini-batches
            for (int iStartMiniBatch = 0; iStartMiniBatch < dataSet.Size; iStartMiniBatch += network.MiniBatchSize)  
            {

                // Feed a mini-batch to the network
                for (int iMiniBatchItem = 0; iMiniBatchItem < network.MiniBatchSize; iMiniBatchItem++)
                {
                    miniBatchItems[iMiniBatchItem] = iStartMiniBatch + iMiniBatchItem;
                }
                network.FeedData(dataSet, miniBatchItems);
                    
                // Run network forward
                network.ForwardPass();

                // Find maximum output score (i.e. assigned class) of each mini batch item
                for (int m = 0; m < network.MiniBatchSize; m++)
                {
                    outputScores = network.Layers.Last().OutputClassScores[m];

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

                    assignedLabel = Utils.IndexOfMax(outputScores);
                    trueLabel = dataSet.GetLabel(miniBatchItems[m]);

                    // Cumulate loss and error
                    loss -= Math.Log(outputScores[trueLabel]);
                    error += (assignedLabel == trueLabel) ? 0 : 1;

                }

                iMiniBatch++;
            } // end loop over mini-batches
             
            error /= dataSet.Size;
        }
    }
}
