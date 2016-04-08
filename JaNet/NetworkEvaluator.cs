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
#if OPENCL_ENABLED

        private int inputBufferBytesSize;
        private IntPtr[] classificationGlobalWorkSize;
        private IntPtr[] classificationLocalWorkSize;

        private Mem assignedClassBuffer;

        private ErrorCode clError;
        private Event clEvent;
#endif
        #endregion
        

        #region Properties

        #endregion


        #region Constructor

        public NetworkEvaluator(int dataDimension, int nClasses, int miniBatchSize)
        {
#if OPENCL_ENABLED
            this.inputBufferBytesSize = sizeof(float) * dataDimension;
            this.classificationGlobalWorkSize = new IntPtr[] { (IntPtr)(miniBatchSize * nClasses) };
            this.classificationLocalWorkSize = new IntPtr[] { (IntPtr)(nClasses) };

            this.clError = new ErrorCode();
            this.clEvent = new Event();

            this.assignedClassBuffer = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(int) * miniBatchSize),
                                                            out clError);
            OpenCLSpace.CheckErr(clError, "NetworkEvaluator.SetupCLObjects: Cl.CreateBuffer assignedClassBuffer");
#endif
        }
        #endregion


        #region Methods

        public void ComputeLossError(NeuralNetwork network, DataSet dataSet, out double loss, out double error)
        {
            loss = 0.0;
            error = 0.0;

            //TODO: do this using OpenCL

            float[] outputScores = new float[dataSet.NumberOfClasses];
            int assignedClass;
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
                clError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                network.Layers.Last().Output.ActivationsGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)outputBufferBytesSize,
                                                outputScores,  // destination
                                                0,
                                                null,
                                                out clEvent);
                OpenCLSpace.CheckErr(clError, "NetworkEvaluator.ComputeCost Cl.clEnqueueReadBuffer outputScores");

                clError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(clError, "Cl.Finish");

                clError = Cl.ReleaseEvent(clEvent);
                OpenCLSpace.CheckErr(clError, "Cl.ReleaseEvent");
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

                
                assignedClass = Utils.IndexOfMax(outputScores);

                // Cumulate loss and error
                loss -= Math.Log(outputScores[assignedClass]);
                error += (assignedClass == dataSet.GetLabel(i)) ? 0 : 1;
                
            }
   
            error /= dataSet.Size;
        }

        #endregion
    }
}
