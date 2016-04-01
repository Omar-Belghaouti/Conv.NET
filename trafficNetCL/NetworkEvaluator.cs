using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{

    static class NetworkEvaluator
    {

#if OPENCL_ENABLED
        
        private static int inputBufferBytesSize;
        private static IntPtr[] classificationGlobalWorkSize;
        private static IntPtr[] classificationLocalWorkSize;
#endif

#if OPENCL_ENABLED
        public static void SetupCLObjects(DataSet dataSet, int miniBatchSize)
        {
            inputBufferBytesSize = sizeof(float) * dataSet.GetDataPoint(0).Length;
            classificationGlobalWorkSize = new IntPtr[] { (IntPtr)(miniBatchSize * dataSet.NumberOfClasses) };
            classificationLocalWorkSize = new IntPtr[] { (IntPtr)(dataSet.NumberOfClasses) };
        }
#endif

        /*
         * 
        private static double classificationError;

        public static double ClassificationError
        {
            get { return classificationError; }
        }

        
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
        */

        public static double ComputeClassificationError(NeuralNetwork network, DataSet dataSet)
        {
            double classificationError = 0;

            // FORWARD PASS (all data points)

            for (int i = 0; i < dataSet.Size; i++)
            {
#if OPENCL_ENABLED
                //TODO: generalise to miniBatchSize > 1
                //TODO: fix
                network.ForwardPass(dataSet.DataGPU(i), inputBufferBytesSize);
#else
                //TODO: generalise to miniBatchSize > 1
                network.ForwardPass(dataSet.GetDataPoint(i));
#endif

                // COUNT CLASSIFICATION ERRORS

#if OPENCL_ENABLED

                Mem assignedClassBuffer = (Mem)Cl.CreateBuffer( CL.Context, 
                                                                MemFlags.ReadWrite, 
                                                                (IntPtr)(sizeof(int) * NetworkTrainer.MiniBatchSize), 
                                                                out CL.Error);
                CL.CheckErr(CL.Error, "NetworkEvaluator.ComputeClassificationError: Cl.CreateBuffer assignedClassBuffer");

                CL.Error  = Cl.SetKernelArg(CL.CheckClassification, 0, assignedClassBuffer);
                CL.Error |= Cl.SetKernelArg(CL.CheckClassification, 1, network.Layers.Last().Output.ActivationsGPU);
                //CL.Error |= Cl.SetKernelArg(CL.CheckClassification, 2, dataSet.LabelsGPU(i));
                CL.Error |= Cl.SetKernelArg(CL.CheckClassification, 2, (IntPtr)sizeof(int), dataSet.NumberOfClasses);
                CL.CheckErr(CL.Error, "NetworkEvaluator.ComputeClassificationError: Cl.SetKernelArg");

                CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue,
                                                    CL.CheckClassification,
                                                    1,
                                                    null,
                                                    classificationGlobalWorkSize,
                                                    classificationLocalWorkSize,
                                                    0,
                                                    null,
                                                    out CL.Event);
                CL.CheckErr(CL.Error, "NetworkEvaluator.ComputeClassificationError: Cl.EnqueueNDRangeKernel");

                int assignedClass = new int();
                CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                assignedClassBuffer, // source
                                                Bool.True, 
                                                (IntPtr)0, 
                                                (IntPtr)sizeof(int),
                                                assignedClass, 
                                                0, 
                                                null, 
                                                out CL.Event);
                CL.CheckErr(CL.Error, "NetworkEvaluator.ComputeClassificationError: Cl.clEnqueueReadBuffer assignedClass");


                /* ------------------------- DEBUGGING ---------------------------------------------

                // Display assigned class and correct class

                int assignedClass = new int();
                float[] classScores = new float[dataSet.NumberOfClasses];
                CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                network.Layers.Last().Output.ActivationsGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(dataSet.NumberOfClasses * sizeof(float)),
                                                classScores,  // destination
                                                0,
                                                null,
                                                out CL.Event);
                CL.CheckErr(CL.Error, "NetworkEvaluator.ComputeClassificationError Cl.clEnqueueReadBuffer classScores");

                assignedClass = Utils.IndexOfMax(classScores);

                Console.WriteLine("\nData point {0}: \n\tAssigned class = {1}\n\tTrue class = {2}", i, assignedClass, dataSet.GetLabel(i));
                Console.ReadKey();


                ------------------------- END DEBUGGING --------------------------------------------- */


                classificationError += (assignedClass == dataSet.GetLabel(i)) ? 0 : 1;

#else

                int outputClassMaxScore = Utils.IndexOfMax(network.Layers.Last().Output.GetHost());
                if (outputClassMaxScore != dataSet.GetLabel(i))
                    classificationError += 1;
#endif
            }

            return classificationError / dataSet.Size;
        }


    }
}
