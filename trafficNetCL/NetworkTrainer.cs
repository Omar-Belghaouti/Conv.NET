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

        private static double errorTraining;
        private static double errorValidation;

        
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

        public static double ErrorTraining
        {
            get { return errorTraining; }
        }

        public static double ErrorValidation
        {
            get { return errorValidation; }
        }

        #endregion


        /// <summary>
        /// Train a neural network using given data.
        /// </summary>
        /// <param name="net"></param>
        /// <param name="trainingSet"></param>
        /// <param name="validationSet"></param>
        /// <returns></returns>
        public static int Train(NeuralNetwork Network, DataSet TrainingSet, DataSet ValidationSet)
        {
            int errorCode = 0;
            bool stopFlag = false;
            int epoch = 0;
            do
            {
                errorCode = TrainOneEpoch(Network, TrainingSet, ValidationSet, out errorTraining, out errorValidation);

                if (errorTraining < errorTolerance)
                    stopFlag = true;

                // TO-DO: also implement early stopping (stop if validation error starts increasing)
                epoch++;

            } while (epoch < maxTrainingEpochs && !stopFlag);

            return errorCode; // error code
        }

        static int TrainOneEpoch(NeuralNetwork Network, DataSet TrainingSet, DataSet ValidationSet,
                    out double errorTraining, out double errorValidation)
        {
            int errorCode = 0;

            Debug.Assert(TrainingSet.Size % miniBatchSize == 0);
            int nMiniBatches = TrainingSet.Size / miniBatchSize;

            // TO-DO: split training set into mini-batches

            // TO-DO: implement single-epoch training

            // At the end of the epoch we should get a training error and a validation error
            //... compute online or "test" with whole training/validation sets??
            errorTraining = 1;
            errorValidation = 1;


            return errorCode;
        }



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

                        // FORWARD PASS
#if OPENCL_ENABLED
                        //TODO: generalise to miniBatchSize > 1
                        network.ForwardPass(trainingSet.DataGPU(iDataPoint), inputBufferBytesSize);
#else
                        //TODO: generalise to miniBatchSize > 1
                        network.ForwardPass(trainingSet.GetDataPoint(iDataPoint));
#endif
                        // COMPUTE ERROR AND GRADIENT

#if OPENCL_ENABLED
                        //TODO: implement this
#else
                        outputScores = network.Layers.Last().Output.GetHost();
                        labelArray = trainingSet.GetLabelArray(iDataPoint);

                        // Gradient of quadratic cost, using LINQ
                        // network.Layers.Last().Output.Delta = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();
                        // network.Layers.Last().BackPropOneCPU();

                        // Gradient of cross-entropy cost (directly write in INPUT delta)
                        network.Layers.Last().Input.DeltaHost = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();

#endif

                        // BACKWARD PASS


                        // Now run backwards and update deltas (cumulating them), but DO NOT update parameters
                        for (int l = network.Layers.Count - 2; l >= 0; l--) // propagate deltas in all layers (but the last) backwards (L-2 to 0)
                        {
                            network.Layers[l].BackPropagate();
                            network.Layers[l].UpdateParameters(learningRate, momentumMultiplier);
                        }

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


        
        public static double TrainMNIST(NeuralNetwork network, DataSet trainingSet)
        {
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network training");
            Console.WriteLine("=========================================\n");
            

            // Initializations
            int nLayers = network.NumberOfLayers;
            int[] randomIntSequence = new int[trainingSet.Size];
            int iDataPoint;
            bool stopFlag = false;
            double errorEpoch;
            bool isOutputEpoch = true;
            int epochsRemainingToOutput = 0;

#if OPENCL_ENABLED


            // Create output buffer
            int inputBufferBytesSize = sizeof(float) * trainingSet.GetDataPoint(0).Length;
            IntPtr outputBufferBytesSizePtr = (IntPtr)(sizeof(float) * trainingSet.NumberOfClasses);
            Mem outputScoresGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, outputBufferBytesSizePtr, out CL.Error);
            CL.CheckErr(CL.Error, "Cl.CreateBuffer outputScoresGPU");

            // Declare work group size for gradient kernel
            IntPtr[] gradientGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(miniBatchSize * trainingSet.NumberOfClasses) }; 
            IntPtr[] gradientLocalWorkSizePtr = new IntPtr[] { (IntPtr)(trainingSet.NumberOfClasses) };

            // Setup evaluator accordingly
            NetworkEvaluator.SetupCLObjects(trainingSet, miniBatchSize);

#else
            float[] outputScores = new float[trainingSet.NumberOfClasses];
            float[] labelArray = new float[trainingSet.NumberOfClasses];
#endif

            int epoch = 0;

            Stopwatch stopwatch = Stopwatch.StartNew();
            do // loop over training epochs
            {
                randomIntSequence = Utils.GenerateRandomPermutation(trainingSet.Size);  // newly generated at every epoch

                // Run over mini-batches
                for (int iStartMiniBatch = 0; iStartMiniBatch < trainingSet.Size; iStartMiniBatch += miniBatchSize) //  
                {
                    // Run over a mini-batch
                    for (int iWithinMiniBatch = 0; iWithinMiniBatch < miniBatchSize; iWithinMiniBatch++)
                    {
                        iDataPoint = randomIntSequence[iStartMiniBatch + iWithinMiniBatch];

                        // FORWARD PASS

#if OPENCL_ENABLED
                        //TODO: generalise to miniBatchSize > 1
                        network.ForwardPass(trainingSet.DataGPU(iDataPoint), inputBufferBytesSize);
#else
                        //TODO: generalise to miniBatchSize > 1
                        network.ForwardPass(trainingSet.GetDataPoint(iDataPoint));
#endif

                        // COMPUTE ERROR

                        /*
#if OPENCL_ENABLED
                        // Set kernel arguments
                        CL.Error  = Cl.SetKernelArg(CL.CrossEntropyGradient, 0, network.Layers[nLayers-1].Input.DeltaGPU);
                        CL.Error |= Cl.SetKernelArg(CL.CrossEntropyGradient, 1, network.Layers[nLayers-1].Output.ActivationsGPU);
                        CL.Error |= Cl.SetKernelArg(CL.CrossEntropyGradient, 2, trainingSet.LabelArraysGPU(iDataPoint));
                        CL.CheckErr(CL.Error, "TrainMNIST.CrossEntropyGradient: Cl.SetKernelArg");
                        //Debugger.Launch();

                        // Run kernel
                        CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue, 
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
                        outputScores = network.Layers.Last().Output.GetHost();
                        labelArray = trainingSet.GetLabelArray(iDataPoint);

                        network.Layers.Last().Input.DeltaHost = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();
#endif

                        // BACKWARD PASS
                        
                        for (int l = nLayers - 2; l >= 0; l--) // propagate deltas in all layers (but the last) backwards (L-2 to 0)
                        {
                            //network.Layers[l].BackPropagate();
                            //network.Layers[l].UpdateParameters(learningRate, momentumMultiplier);
                        }
                         */ 

                    } // end loop over mini-batches

                }

                
                isOutputEpoch = epochsRemainingToOutput == 0;
                if (isOutputEpoch)
                {
                    errorEpoch = NetworkEvaluator.ComputeClassificationError(network, trainingSet);
                    Console.WriteLine("Epoch {0}: classification error = {1}", epoch, errorEpoch);

                    if (errorEpoch < errorTolerance)
                        stopFlag = true;

                    Console.WriteLine("Epoch time: {0} ms", stopwatch.ElapsedMilliseconds);
                    stopwatch.Restart();

                    epochsRemainingToOutput = consoleOutputLag;
                    isOutputEpoch = false;
                }
                epochsRemainingToOutput--;
                
                
                // TO-DO: also implement early stopping (stop if validation error starts increasing)
                epoch++;

                

            } while (epoch < maxTrainingEpochs && !stopFlag);

            stopwatch.Stop();

            return NetworkEvaluator.ComputeClassificationError(network, trainingSet);
        }






        /// <summary>
        /// Cross-entropy cost for a single example
        /// </summary>
        /// <param name="targetValues"></param>
        /// <param name="networkOutputs"></param>
        /// <param name="gradient"></param>
        /// <returns></returns>
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
