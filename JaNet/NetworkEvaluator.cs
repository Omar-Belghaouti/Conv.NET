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

#endif
        #endregion

        #region Properties

        #endregion


#if OPENCL_ENABLED
        public void SetupCL(int dataDimension, int nClasses, int miniBatchSize)
        {
            inputBufferBytesSize = sizeof(float) * dataDimension;
            classificationGlobalWorkSize = new IntPtr[] { (IntPtr)(miniBatchSize * nClasses) };
            classificationLocalWorkSize = new IntPtr[] { (IntPtr)(nClasses) };

            assignedClassBuffer = (Mem) Cl.CreateBuffer(CL.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(int) * miniBatchSize),
                                                        out CL.Error);
            CL.CheckErr(CL.Error, "NetworkEvaluator.SetupCLObjects: Cl.CreateBuffer assignedClassBuffer");
        }
#endif


        public void ComputeLossError(NeuralNetwork network, DataSet dataSet, out double loss, out double error)
        {
            // pass network as argument if it doesn't work!

            loss = 0.0;
            error = 0.0;

            float[] outputScores = new float[dataSet.NumberOfClasses];
            int assignedClass;
            int outputBufferBytesSize = dataSet.NumberOfClasses * sizeof(float);

            // loop through all data points in dataSet
            for (int i = 0; i < dataSet.Size; i++)
            {
                // Feed input data
                network.FeedData(dataSet, i);

                // Run network forward
                network.ForwardPass();

                
                // Find maximum output score (i.e. assigned class)
                
#if OPENCL_ENABLED
                CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                network.Layers.Last().Output.ActivationsGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)outputBufferBytesSize,
                                                outputScores,  // destination
                                                0,
                                                null,
                                                out CL.Event);
                CL.CheckErr(CL.Error, "NetworkEvaluator.ComputeCost Cl.clEnqueueReadBuffer outputScores");

                CL.Error = Cl.Finish(CL.Queue);
                CL.CheckErr(CL.Error, "Cl.Finish");

                CL.Error = Cl.ReleaseEvent(CL.Event);
                CL.CheckErr(CL.Error, "Cl.ReleaseEvent");
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
    }
}
