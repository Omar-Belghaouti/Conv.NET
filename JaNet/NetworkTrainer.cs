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
        #region Fields

        // Neural network to train
        private NeuralNetwork network;

        // Data
        private DataSet trainingSet;
        private DataSet validationSet;
        private int nClasses;

        // Hyperparameters
        private double learningRate;
        private double momentumMultiplier;
        private int maxTrainingEpochs;
        private int miniBatchSize;
        private double errorTolerance;
        private int consoleOutputLag;
        private bool evaluateBeforeTraining;
        private bool earlyStopping;

        // Losses/Errors
        private double lossTraining;
        private double lossValidation;
        private double errorTraining = 1.0;
        private double errorValidation = 1.0;

#if OPENCL_ENABLED
        // Gradient kernel work sizes 
        private IntPtr[] gradientGlobalWorkSizePtr; 
        private IntPtr[] gradientLocalWorkSizePtr;
#endif

        #endregion


        #region Properties

        public NeuralNetwork Network
        {
            get { return network; }
        }

        public double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }

        public double MomentumMultiplier
        {
            get { return momentumMultiplier; }
            set { momentumMultiplier = value; }
        }

        public int MaxTrainingEpochs
        {
            get { return maxTrainingEpochs; }
            set { maxTrainingEpochs = value; }
        }

        public int MiniBatchSize
        {
            get { return miniBatchSize; }
            set { miniBatchSize = value; }
        }

        public double ErrorTolerance
        {
            get { return errorTolerance; }
            set { errorTolerance = value; }
        }

        public int ConsoleOutputLag
        {
            get { return consoleOutputLag; }
            set { consoleOutputLag = value; }
        }

        public double LossTraining
        {
            get { return lossTraining; }
        }

        public double LossValidation
        {
            get { return lossValidation; }
        }

        public double ErrorTraining
        {
            get { return errorTraining; }
        }

        public double ErrorValidation
        {
            get { return errorValidation; }
        }

        public bool EvaluateBeforeTraining
        {
            set { evaluateBeforeTraining = value; }
        }

        public bool EarlyStopping
        {
            set { earlyStopping = value; }
        }

        #endregion


        #region Constructor

        public NetworkTrainer(NeuralNetwork Network, DataSet TrainingSet, DataSet ValidationSet)
        {
            if (Network == null)
                throw new ArgumentException("Cannot train an empty network!");

            if (ValidationSet != null)
            {
                if (TrainingSet.NumberOfClasses != ValidationSet.NumberOfClasses)
                    throw new ArgumentException("Number of classes in training and validation set are different!");

                this.validationSet = ValidationSet;
            }

            this.nClasses = TrainingSet.NumberOfClasses;
            this.network = Network;
            this.trainingSet = TrainingSet;



#if OPENCL_ENABLED
            // Set global and local work sizes
            gradientGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(nClasses) };
            gradientLocalWorkSizePtr = new IntPtr[] { (IntPtr)(nClasses) };
#endif
        }

        #endregion


        public void Train()
        {
            int epoch = 0;
            bool isOutputEpoch = true;
            int epochsRemainingToOutput = (evaluateBeforeTraining == true) ? 0 : consoleOutputLag;

            Sequence indicesSequence = new Sequence(trainingSet.Size);
            int iDataPoint;

            bool stopFlag = false;

            NetworkEvaluator networkEvaluator = new NetworkEvaluator(trainingSet.GetDataPoint(0).Length, nClasses, miniBatchSize);

            // Timing
            Stopwatch stopwatch = Stopwatch.StartNew();
            Stopwatch stopwatchFwd = Stopwatch.StartNew();
            Stopwatch stopwatchGrad = Stopwatch.StartNew();
            Stopwatch stopwatchBwd = Stopwatch.StartNew();

            while (epoch < maxTrainingEpochs && !stopFlag) // loop over training epochs
            {

                /********************
                 * Console output
                 *******************/

                isOutputEpoch = epochsRemainingToOutput == 0;
                if (isOutputEpoch)
                {
                    Console.WriteLine("\nTime to evaluate the network!");

                    // Evaluate all training set
                    Console.WriteLine("Evaluating on TRAINING set...");
                    stopwatch.Restart();
                    networkEvaluator.ComputeLossError(network, trainingSet, out lossTraining, out errorTraining);
                    Console.WriteLine("\tLoss = {0}\n\tError = {1}\n\tEval runtime = {2}ms\n",
                                        lossTraining, errorTraining, stopwatch.ElapsedMilliseconds);

                    // Evaluate all validation set
                    if (validationSet != null)
                    {
                        double tmpErrorValidation;
                        Console.WriteLine("Evaluating on VALIDATION set...");
                        stopwatch.Restart();
                        networkEvaluator.ComputeLossError(network, validationSet, out lossValidation, out tmpErrorValidation);
                        Console.WriteLine("\tLoss = {0}\n\tError = {1}\n\tEval runtime = {2}ms\n",
                                            lossValidation, tmpErrorValidation, stopwatch.ElapsedMilliseconds);
                        if (tmpErrorValidation < errorValidation)
                        {
                            errorValidation = tmpErrorValidation;
                            // TODO: save a (deep) copy of current network, with all parameters
                        }
                        else
                        {
                            Console.WriteLine("Classification error on the validation set started increasing. Stopping training.");
                            stopFlag = true;
                            break;
                        }

                        if (errorValidation < errorTolerance)
                        {
                            Console.WriteLine("Classification error is below tolerance. Stopping training.");
                            stopFlag = true;
                            break;
                        }
                    }

                    epochsRemainingToOutput = consoleOutputLag;
                    isOutputEpoch = false;
                }

                /********************
                 * Training epoch
                 *******************/

                stopwatch.Restart();

                epochsRemainingToOutput--;

                indicesSequence.Shuffle(); // shuffle at every epoch


                // TODO: implement mini-batch training
                /* 
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
                */
                stopwatchFwd.Reset();
                stopwatchGrad.Reset();
                stopwatchBwd.Reset();

                // Online training:
                for (int i = 0; i < trainingSet.Size; i++)
                {
                    iDataPoint = indicesSequence[i]; // random order

                    // Feed input data (datum, actually)
                    network.FeedData(trainingSet, iDataPoint);

                    // Forward pass
                    stopwatchFwd.Start();
                    network.ForwardPass();
                    stopwatchFwd.Stop();

                    // Compute gradient
                    stopwatchGrad.Start();
                    CrossEntropyGradient(iDataPoint);
                    stopwatchGrad.Stop();

                    // Backpropagate gradient and update parameters
                    stopwatchBwd.Start();
                    network.BackwardPass(learningRate, momentumMultiplier);
                    stopwatchBwd.Stop();
                }

                Console.WriteLine("\nEpoch {0} - Training runtime = {1}ms", epoch, stopwatch.ElapsedMilliseconds);

                Console.WriteLine("Forward: {0}ms - Gradient: {1}ms - Backward: {2}ms",
                    stopwatchFwd.ElapsedMilliseconds, stopwatchGrad.ElapsedMilliseconds, stopwatchBwd.ElapsedMilliseconds);

                epoch++;
            }

            stopwatch.Stop();
        }



        private void CrossEntropyGradient(int iDataPoint)
        {
#if OPENCL_ENABLED

            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.CrossEntropyGradient, 0, network.Layers[network.NumberOfLayers - 1].InputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CrossEntropyGradient, 1, network.Layers[network.NumberOfLayers - 1].OutputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CrossEntropyGradient, 2, trainingSet.LabelArraysGPU(iDataPoint));
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "TrainMNIST.CrossEntropyGradient: Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel( OpenCLSpace.Queue,
                                                OpenCLSpace.CrossEntropyGradient,
                                                1,
                                                null,
                                                gradientGlobalWorkSizePtr,
                                                gradientLocalWorkSizePtr,
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "TrainMNIST.CrossEntropyGradient: Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
            float[] outputScores = network.Layers.Last().Output.GetHost();
            float[] labelArray = dataSet.GetLabelArray(iDataPoint);

            network.Layers.Last().Input.DeltaHost = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();
#endif
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

        #region Junk
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
        #endregion

    }
}
