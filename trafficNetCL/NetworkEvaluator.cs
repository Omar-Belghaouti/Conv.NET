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

        private static Mem assignedClassBuffer;

#endif

#if OPENCL_ENABLED
        public static void SetupCLObjects(DataSet dataSet, int miniBatchSize)
        {
            inputBufferBytesSize = sizeof(float) * dataSet.GetDataPoint(0).Length;
            classificationGlobalWorkSize = new IntPtr[] { (IntPtr)(miniBatchSize * dataSet.NumberOfClasses) };
            classificationLocalWorkSize = new IntPtr[] { (IntPtr)(dataSet.NumberOfClasses) };

            assignedClassBuffer = (Mem)Cl.CreateBuffer( CL.Context, 
                                                        MemFlags.ReadWrite, 
                                                        (IntPtr)(sizeof(int) * NetworkTrainer.MiniBatchSize), 
                                                        out CL.Error);
            CL.CheckErr(CL.Error, "NetworkEvaluator.SetupCLObjects: Cl.CreateBuffer assignedClassBuffer");
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

            // loop through all data points in dataSet
            for (int i = 0; i < dataSet.Size; i++)
            {
                // FEED INPUT DATA

#if OPENCL_ENABLED

                // Copy by reference
                network.Layers[0].Input.ActivationsGPU = dataSet.DataGPU(i);


                // Copy data point in input buffer of the first layer (by value)
                /*
                Cl.EnqueueCopyBuffer(   CL.Queue,
                                        dataSet.DataGPU(i),        // source
                                        network.Layers[0].Input.ActivationsGPU, // destination
                                        (IntPtr)null,
                                        (IntPtr)null,
                                        (IntPtr)inputBufferBytesSize,
                                        0,
                                        null,
                                        out CL.Event);
                CL.CheckErr(CL.Error, "NetworkEvaluator: Cl.EnqueueCopyBuffer inputData");
                 * */
#else
                network.Layers[0].Input.SetHost(dataSet.GetDataPoint(i));
#endif
                

                // FORWARD PASS
                
                network.ForwardPass();

                // COUNT CLASSIFICATION ERRORS

#if OPENCL_ENABLED


                // TODO: fix kernel for checking correct classification
                // (done on the host at the moment)

                /*
                CL.Error  = Cl.SetKernelArg(CL.CheckClassification, 0, assignedClassBuffer);
                CL.Error |= Cl.SetKernelArg(CL.CheckClassification, 1, network.Layers.Last().Output.ActivationsGPU);
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
                */


                
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

                int assignedClass = Utils.IndexOfMax(classScores);

                //Console.WriteLine("\nData point {0}: \n\tAssigned class = {1}\n\tTrue class = {2}", i, assignedClass, dataSet.GetLabel(i));
                //Console.ReadKey();


                /* ------------------------- END DEBUGGING --------------------------------------------- */


                /* ------------------------- DEBUGGING --------------------------------------------- */

                //if (assignedClass == dataSet.GetLabel(i))
                //    correctClassifications += (i.ToString() + " ");

                /* ------------------------- END DEBUGGING --------------------------------------------- */


                

#else
                float[] classScores = network.Layers.Last().Output.GetHost();
                int assignedClass = Utils.IndexOfMax(classScores);
                

#endif
                //Console.WriteLine("\nData point {0}: \n\tAssigned class = {1}\n\tTrue class = {2}", i, assignedClass, dataSet.GetLabel(i));
                //Console.ReadKey();
                //Console.WriteLine("\nScores:  {0}  {1}", classScores[0], classScores[1]);
                //Console.ReadKey();
                classificationError += (assignedClass == dataSet.GetLabel(i)) ? 0 : 1;
            }

            //Console.WriteLine("Data points correctly classified: \n" + correctClassifications);
            //Console.ReadKey();

            return classificationError / dataSet.Size;
        }


    }
}
