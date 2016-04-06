using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    class NetworkTrainer
    {
        #region NetworkTrainer fields

        private static double learningRate;
        private static double momentumMultiplier;
        private static int maxTrainingEpochs;
        private static int miniBatchSize;
        private static double errorTolerance;
        private static int consoleOutputLag;

        //private static double errorTraining;
        //private static double errorValidation;

#if OPENCL_ENABLED
        // Global and local work group size for gradient kernel
        private static IntPtr[] gradientGlobalWorkSizePtr; 
        private static IntPtr[] gradientLocalWorkSizePtr; 
#endif

        
        #endregion


        #region NetworkTrainer properties

        public static double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }

        public static double MomentumMultiplier
        {
            get { return momentumMultiplier; }
            set { momentumMultiplier = value; }
        }

        public static int MaxTrainingEpochs
        {
            get { return maxTrainingEpochs; }
            set { maxTrainingEpochs = value; }
        }

        public static int MiniBatchSize
        {
            get { return miniBatchSize; }
            set { miniBatchSize = value; }
        }

        public static double ErrorTolerance
        {
            get { return errorTolerance; }
            set { errorTolerance = value; }
        }

        public static int ConsoleOutputLag
        {
            get { return consoleOutputLag; }
            set { consoleOutputLag = value; }
        }

        /*
        public static double ErrorTraining
        {
            get { return errorTraining; }
        }

        public static double ErrorValidation
        {
            get { return errorValidation; }
        }
        */
        #endregion

#if OPENCL_ENABLED
        public static void SetupCL(DataSet dataSet)
        {
            gradientGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(miniBatchSize * dataSet.NumberOfClasses) }; 
            gradientLocalWorkSizePtr = new IntPtr[] { (IntPtr)(dataSet.NumberOfClasses) };
        }
#endif


#if TOY
        /// <summary>
        /// Training on toy 2D data set (to test network structure, fully connected, tanh, softmax and respective backprops)
        /// </summary>
        /// <param name="network"></param>
        /// <param name="trainingSet"></param>
        /// <param name="finalError"></param>
        /// <param name="finalEpoch"></param>
        /// <returns></returns>
        public static void TrainSimpleTest(NeuralNetwork network, DataSet trainingSet)
        {

            // Initializations
            int nLayers = network.NumberOfLayers;
            int[] randomIntSequence = new int[trainingSet.Size];
            int iDataPoint;
            bool stopFlag = false;
            
            double errorEpoch;
            bool isOutputEpoch = true;
            int epochsRemainingToOutput = 0;
            List<int[]> miniBatchList = new List<int[]>();
            int nMiniBatches = trainingSet.Size / miniBatchSize;
            float[] outputScores = new float[trainingSet.NumberOfClasses];
            float[] labelArray = new float[trainingSet.NumberOfClasses];

#if OPENCL_ENABLED
            int inputBufferBytesSize = sizeof(float) * trainingSet.GetDataPoint(0).Length;

            // Global and local work group size for gradient kernel
            IntPtr[] gradientGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(miniBatchSize * trainingSet.NumberOfClasses) };
            IntPtr[] gradientLocalWorkSizePtr = new IntPtr[] { (IntPtr)(trainingSet.NumberOfClasses) };


#endif

            int epoch = 0;
            do // loop over training epochs
            {
                randomIntSequence = Utils.GenerateRandomPermutation(trainingSet.Size);  // new every epoch

                // Run over mini-batches
                for (int iStartMiniBatch = 0; iStartMiniBatch < trainingSet.Size; iStartMiniBatch += miniBatchSize)
                {
                    // Run over a mini-batch
                    for (int iWithinMiniBatch = 0; iWithinMiniBatch < miniBatchSize; iWithinMiniBatch++) 
                    {
                        iDataPoint = randomIntSequence[iStartMiniBatch + iWithinMiniBatch];

                        // FEED INPUT DATA
#if OPENCL_ENABLED

                        // Feed by reference
                        network.Layers[0].Input.ActivationsGPU = trainingSet.DataGPU(iDataPoint);

                        // Copy data point in input buffer of the first layer
                        /*
                        Cl.EnqueueCopyBuffer(CL.Queue,
                                                trainingSet.DataGPU(iDataPoint),        // source
                                                network.Layers[0].Input.ActivationsGPU, // destination
                                                (IntPtr)null,
                                                (IntPtr)null,
                                                (IntPtr)inputBufferBytesSize,
                                                0,
                                                null,
                                                out CL.Event);
                        CL.CheckErr(CL.Error, "NetworkTrainer.TrainSimpleTest: Cl.EnqueueCopyBuffer inputData");
                         */
#else
                        network.Layers[0].Input.SetHost(trainingSet.GetDataPoint(iDataPoint));
#endif



                        // FORWARD PASS
                        network.ForwardPass();

                        // COMPUTE ERROR AND GRADIENT
#if OPENCL_ENABLED
                        // Set kernel arguments
                        CL.Error  = Cl.SetKernelArg(CL.CrossEntropyGradient, 0, network.Layers[nLayers - 1].Input.DeltaGPU);
                        CL.Error |= Cl.SetKernelArg(CL.CrossEntropyGradient, 1, network.Layers[nLayers - 1].Output.ActivationsGPU);
                        CL.Error |= Cl.SetKernelArg(CL.CrossEntropyGradient, 2, trainingSet.LabelArraysGPU(iDataPoint));
                        CL.CheckErr(CL.Error, "TrainSimpleTest.CrossEntropyGradient: Cl.SetKernelArg");

                        // Run kernel
                        CL.Error = Cl.EnqueueNDRangeKernel(CL.Queue,
                                                            CL.CrossEntropyGradient,
                                                            1,
                                                            null,
                                                            gradientGlobalWorkSizePtr,
                                                            gradientLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out CL.Event);
                        CL.CheckErr(CL.Error, "TrainSimpleTest.CrossEntropyGradient: Cl.EnqueueNDRangeKernel");
#else
                        outputScores = network.Layers.Last().Output.GetHost();
                        labelArray = trainingSet.GetLabelArray(iDataPoint);

                        // Gradient of cross-entropy cost (directly write in INPUT delta)
                        network.Layers.Last().Input.DeltaHost = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();

#endif
                        
#if DEBUGGING_STEPBYSTEP
                        /* ------------------------- DEBUGGING --------------------------------------------- */
                        // Display output activation
#if OPENCL_ENABLED
                        float[] outputScoresGPU = new float[network.Layers[nLayers - 1].Output.NumberOfUnits];
                        CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                        network.Layers[nLayers - 1].Output.ActivationsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(network.Layers[nLayers - 1].Output.NumberOfUnits * sizeof(float)),
                                                        outputScoresGPU,  // destination
                                                        0,
                                                        null,
                                                        out CL.Event);
                        CL.CheckErr(CL.Error, "NetworkTrainer Cl.clEnqueueReadBuffer outputScoresGPU");

                        Console.WriteLine("\nOutput scores:");
                        for (int j = 0; j < outputScoresGPU.Length; j++)
                            Console.Write("{0}  ", outputScoresGPU[j]);
                        Console.WriteLine();
#else
                        Console.WriteLine("\nOutput scores:");
                        for (int j = 0; j < outputScores.Length; j++)
                            Console.Write("{0}  ", outputScores[j]);
                        Console.WriteLine();
#endif
                        /* ------------------------- END --------------------------------------------- */
#endif


#if DEBUGGING_STEPBYSTEP
                        /* ------------------------- DEBUGGING --------------------------------------------- */

                        // Display true data label CPU

                        float[] labelArrayHost = new float[trainingSet.NumberOfClasses];
                        labelArrayHost = trainingSet.GetLabelArray(iDataPoint);

                        Console.WriteLine("\nData label array on HOST:");
                        for (int j = 0; j < labelArrayHost.Length; j++)
                            Console.Write("{0}  ", labelArrayHost[j]);
                        Console.WriteLine();
                        /* ------------------------- END --------------------------------------------- */
#endif

#if DEBUGGING_STEPBYSTEP
                        /* ------------------------- DEBUGGING --------------------------------------------- */
                        // Display true data label
#if OPENCL_ENABLED
                        float[] labelArrayGPU = new float[trainingSet.NumberOfClasses];
                        CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                        trainingSet.LabelArraysGPU(iDataPoint), // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(trainingSet.NumberOfClasses * sizeof(float)),
                                                        labelArrayGPU,  // destination
                                                        0,
                                                        null,
                                                        out CL.Event);
                        CL.CheckErr(CL.Error, "NetworkTrainer Cl.clEnqueueReadBuffer labelArrayGPU");

                        Console.WriteLine("\nData label array on DEVICE:");
                        for (int j = 0; j < labelArrayGPU.Length; j++)
                            Console.Write("{0}  ", labelArrayGPU[j]);
                        Console.WriteLine();
#endif
                        /* ------------------------- END --------------------------------------------- */
#endif

#if DEBUGGING_STEPBYSTEP
                        /* ------------------------- DEBUGGING --------------------------------------------- */
                        // Display gradient

                        float[] gradient = new float[network.Layers[nLayers - 1].Input.NumberOfUnits];
#if OPENCL_ENABLED
                        CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                        network.Layers[nLayers - 1].Input.DeltaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(network.Layers[nLayers - 1].Input.NumberOfUnits * sizeof(float)),
                                                        gradient,  // destination
                                                        0,
                                                        null,
                                                        out CL.Event);
                        CL.CheckErr(CL.Error, "NetworkTrainer Cl.clEnqueueReadBuffer gradient");
#else
                        gradient = network.Layers.Last().Input.DeltaHost;
#endif
                        Console.WriteLine("\nGradient written to final layer:");
                        for (int j = 0; j < gradient.Length; j++)
                            Console.Write("{0}  ", gradient[j]);
                        Console.WriteLine();
                        Console.ReadKey();


                        /*------------------------- END DEBUGGING --------------------------------------------- */
#endif


                        // BACKWARD PASS (includes parameter updating)

                        network.BackwardPass(learningRate, momentumMultiplier);

                        // TEST: try cleaning stuff

                    } // end loop over mini-batches
                }


                if (isOutputEpoch)
                {
                    //costEpoch = QuadraticCost(network, trainingSet);
                    errorEpoch = NetworkEvaluator.ComputeClassificationError(network, trainingSet);
                    Console.WriteLine("Epoch {0}: classification error = {1}", epoch, errorEpoch);

                    if (errorEpoch < errorTolerance)
                        stopFlag = true;

                    epochsRemainingToOutput = consoleOutputLag;
                    isOutputEpoch = false;
                }
                epochsRemainingToOutput--;
                isOutputEpoch = epochsRemainingToOutput == 0;
                

                // TO-DO: also implement early stopping (stop if validation error starts increasing)
                epoch++;

            } while (epoch < maxTrainingEpochs && !stopFlag);

        }
#endif
        
        public static void TrainMNIST(NeuralNetwork network, DataSet trainingSet)
        {
            // Initializations
            int nLayers = network.NumberOfLayers;
            Sequence indicesSequence = new Sequence(trainingSet.Size);
            int iDataPoint;
            bool stopFlag = false;
            double lossEpoch;
            double errorEpoch;
            bool isOutputEpoch = true;
            int epochsRemainingToOutput = 10000;
            float[] outputScores = new float[trainingSet.NumberOfClasses]; // TODO: generalise to miniBatchSize > 1
            float[] labelArray = new float[trainingSet.NumberOfClasses]; // TODO: generalise to miniBatchSize > 1

            int epoch = 0;

            Stopwatch stopwatch = Stopwatch.StartNew();
            do // loop over training epochs
            {
                isOutputEpoch = epochsRemainingToOutput == 0;
                if (isOutputEpoch)
                {
                    stopwatch.Restart();
                    NetworkEvaluator.ComputeLossError(network, trainingSet, out lossEpoch, out errorEpoch);
                    Console.WriteLine("\n\tLoss = {0}\n\tError = {1}\n\tEval runtime = {2}ms\n", lossEpoch, errorEpoch, stopwatch.ElapsedMilliseconds);

                    if (errorEpoch < errorTolerance)
                    {
                        stopFlag = true;
                        break;
                    }

                    epochsRemainingToOutput = consoleOutputLag;
                    isOutputEpoch = false;
                }
                epochsRemainingToOutput--;


                stopwatch.Restart();

                indicesSequence.Shuffle(); // shuffle at every epoch

                /* IF DEBUGGING FAILS, RESTORE FROM HERE...
                // Run over mini-batches
                for (int iStartMiniBatch = 0; iStartMiniBatch < trainingSet.Size; iStartMiniBatch += miniBatchSize)  
                {
                    // Run over a mini-batch
                    for (int iWithinMiniBatch = 0; iWithinMiniBatch < miniBatchSize; iWithinMiniBatch++)
                    {
                        iDataPoint = indicesSequence[iStartMiniBatch + iWithinMiniBatch];

                        // Feed input data
                        network.FeedData(trainingSet, iDataPoint);

                        // Forward pass
                        network.ForwardPass();

                        // Compute gradient
                        CrossEntropyGradient(network, trainingSet, iDataPoint);

                        // Backpropagate gradient and update parameters
                        network.BackwardPass(learningRate, momentumMultiplier);

                    } // end loop over mini-batches

                }
                */ // TO HERE 



                // Online training
                for (int i = 0; i < trainingSet.Size; i++)
                {
                    iDataPoint = indicesSequence[i];

                    // Feed input data
                    network.FeedData(trainingSet, iDataPoint);

                    // Forward pass
                    network.ForwardPass();

                    // Compute gradient
                    //CrossEntropyGradient(network, trainingSet, iDataPoint);

                    // Backpropagate gradient and update parameters
                    //network.BackwardPass(learningRate, momentumMultiplier);
                }


                Console.WriteLine("Epoch {0} - Runtime = {1}ms", epoch, stopwatch.ElapsedMilliseconds);

                
                
                // TO-DO: also implement early stopping (stop if validation error starts increasing)
                epoch++;

                

            } while (epoch < maxTrainingEpochs && !stopFlag);

            stopwatch.Stop();

        }



        private static void CrossEntropyGradient(NeuralNetwork network, DataSet dataSet, int iDataPoint)
        {
            // Compute gradient
#if OPENCL_ENABLED
            // Set kernel arguments
            CL.Error = Cl.SetKernelArg(CL.CrossEntropyGradient, 0, network.Layers[network.NumberOfLayers - 1].Input.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(CL.CrossEntropyGradient, 1, network.Layers[network.NumberOfLayers - 1].Output.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(CL.CrossEntropyGradient, 2, dataSet.LabelArraysGPU(iDataPoint));
            CL.CheckErr(CL.Error, "TrainMNIST.CrossEntropyGradient: Cl.SetKernelArg");

            // Run kernel
            CL.Error = Cl.EnqueueNDRangeKernel(CL.Queue,
                                                CL.CrossEntropyGradient,
                                                1,
                                                null,
                                                gradientGlobalWorkSizePtr,
                                                gradientLocalWorkSizePtr,
                                                0,
                                                null,
                                                out CL.Event);
            CL.CheckErr(CL.Error, "TrainMNIST.CrossEntropyGradient: Cl.EnqueueNDRangeKernel");
#else
            float[] outputScores = network.Layers.Last().Output.GetHost();
            float[] labelArray = dataSet.GetLabelArray(iDataPoint);
             
            network.Layers.Last().Input.DeltaHost = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();
#endif
        }


        /// <summary>
        /// Cross-entropy cost for a single example
        /// </summary>
        /// <param name="targetValues"></param>
        /// <param name="networkOutputs"></param>
        /// <param name="gradient"></param>
        /// <returns></returns>
        [Obsolete("Now implemented in NetworkEvaluator")]
        static double CrossEntropyCost(float[] targetValues, float[] networkOutputs)
        {
            double cost = 0.0;

            for (int k = 0; k < targetValues.Length; k++)
            {
                cost += targetValues[k] * Math.Log(networkOutputs[k]);
            }

            return cost;
        }


        #region Deprecated methods

        /*

        /// <summary>
        /// Only use for SINGLE OUTPUT UNIT networks! 
        /// </summary>
        /// <param name="network"></param>
        /// <param name="dataSet"></param>
        /// <returns></returns>
        [Obsolete("This method was originally created to deal with the toy 2d example.")]
        static double ClassificationErrorSign(NeuralNetwork network, DataSet dataSet)
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
                int outputClass = Math.Sign(network.Layers.Last().Output.GetHost()[0]);
                classificationError += Math.Abs(outputClass - dataSet.GetLabel(i));
            }

            return classificationError / (2* dataSet.Size);
        }





        
        [Obsolete("Deprecated. Use cross-entropy cost instead.")]
        static double QuadraticCost(float[] targetValues, float[] networkOutputs, out float[] gradient)
        {
            if (targetValues.Length != networkOutputs.Length)
                throw new System.InvalidOperationException("Mismatch between length of output array and target (label) array.");

            gradient = targetValues.Zip(networkOutputs, (x, y) => y - x).ToArray();
            var squaredErrors = gradient.Select(x => Math.Pow(x, 2));

            return squaredErrors.Sum() / squaredErrors.Count();
        }
        

        [Obsolete("Deprecated. Use cross-entropy cost instead.")]
        static double QuadraticCost(NeuralNetwork network, DataSet dataSet)
        {
            float[] dummy;
            double totalCost = 0;

            for (int i = 0; i < dataSet.Size; i++)
            {
                network.Layers[0].Input.SetHost(dataSet.GetDataPoint(i));

                // Run forward
                for (int l = 0; l < network.Layers.Count; l++)
                {
                    network.Layers[l].FeedForward();
                }

                // Compute cost
                totalCost += QuadraticCost(new float[] { (float)dataSet.GetLabel(i) }, network.Layers.Last().Output.GetHost(), out dummy);
            }

            return totalCost / (2 * dataSet.Size);
        }
         * */

        #endregion


    }
}
