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

        public NetworkEvaluator(int dataDimension, int nClasses, int miniBatchSize)
        {
#if OPENCL_ENABLED
            this.inputBufferBytesSize = sizeof(float) * dataDimension;
            this.classificationGlobalWorkSize = new IntPtr[] { (IntPtr)(miniBatchSize * nClasses) };
            this.classificationLocalWorkSize = new IntPtr[] { (IntPtr)(nClasses) };

            this.assignedClassBuffer = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(int) * miniBatchSize),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NetworkEvaluator.SetupCLObjects: Cl.CreateBuffer assignedClassBuffer");
#endif
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

            // loop through all data points in dataSet (one by one)
            // TODO: once implemented in OpenCL, do not loop over data one by one, instead find an optimal batch size to do this faster
            for (int i = 0; i < dataSet.Size; i++)
            {
                // Feed input data
                network.FeedData(dataSet, i);

                // Run network forward
                network.ForwardPass();

                
                // Find maximum output score (i.e. assigned class)
                
#if OPENCL_ENABLED
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                            network.Layers.Last().OutputNeurons.ActivationsGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)outputBufferBytesSize,
                                                            outputScores,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NetworkEvaluator.ComputeCost Cl.clEnqueueReadBuffer outputScores");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
                outputScores = network.Layers.Last().Output.GetHost();
#endif



                /////////////// DEBUGGING (visualize true label vs network output)
                /*
                Console.WriteLine("\n\n\tData point {0}:", i);
                float[] trueLabel = dataSet.GetLabelArray(i);
                Console.WriteLine("\nTrue label:");
                for (int iClass = 0; iClass < dataSet.NumberOfClasses; iClass++)
                {
                    Console.Write( trueLabel[iClass].ToString("0.##") + " ");
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
                trueLabel = dataSet.GetLabel(i);
                // Cumulate loss and error
                loss -= Math.Log(outputScores[trueLabel]);
                error += (assignedLabel == trueLabel) ? 0 : 1;
                
            }
   
            error /= dataSet.Size;
        }
    }
}
