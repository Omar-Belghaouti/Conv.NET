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

        // Hyperparameters
        private double learningRate;
        private double momentumMultiplier;
        private double weightDecayCoeff;
        private int maxTrainingEpochs;
        private int miniBatchSize;
        private double errorTolerance;
        private int consoleOutputLag;
        private bool evaluateBeforeTraining;
        private bool earlyStopping;
        private double dropoutFC;
        private double dropoutConv;
        private int epochsBeforeDropout;

        // Losses/Errors
        private double lossTraining;
        private double lossValidation;
        private double errorTraining;
        private double errorValidation;

        // Paths for saving data
        //private bool saveIters;
        //private string trainingIterSavePath;
        private string trainingEpochSavePath;
        private string validationEpochSavePath;

        #endregion


        #region Properties

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

        public double WeightDecayCoeff
        {
            get { return weightDecayCoeff; }
            set { weightDecayCoeff = value; }
        }

        public int MaxTrainingEpochs
        {
            get { return maxTrainingEpochs; }
            set { maxTrainingEpochs = value; }
        }

        public int MiniBatchSize
        {
            get { return miniBatchSize; }
            set 
            {
#if OPENCL_ENABLED
                if (OpenCLSpace.OPTIMAL_GROUP_SIZE % value != 0)
                    throw new ArgumentException("OPTIMAL_GROUP_SIZE should divide miniBatchSize.");
#endif
                miniBatchSize = value; 
            }
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

        public double DropoutFullyConnected
        {
            get { return dropoutFC; }
            set { dropoutFC = value; }
        }

        public double DropoutConvolutional
        {
            get { return dropoutConv; }
            set { dropoutConv = value; }
        }

        public int EpochsBeforeDropout
        {
            get { return epochsBeforeDropout; }
            set 
            {
                if (epochsBeforeDropout < 0)
                    throw new ArgumentException("Property NetworkTrainer.EpochsBeforeDropout only accepts non-negative integers.");
                else
                    epochsBeforeDropout = value; 
            }
        }

        //public bool SaveIters
        //{
        //    set { saveIters = value; }
        //}

        //public string TrainingIterSavePath
        //{
        //    set { trainingIterSavePath = value; }
        //}

        public string TrainingEpochSavePath
        {
            set { trainingEpochSavePath = value; }
        }

        public string ValidationEpochSavePath
        {
            set { validationEpochSavePath = value; }
        }

        #endregion


        public void Train(NeuralNetwork network, DataSet trainingSet, DataSet validationSet)
        {
            // Setup network before training
            network.Set("MiniBatchSize", this.miniBatchSize);
            network.Set("DropoutConv", this.dropoutConv);
            network.Set("DropoutFC", this.dropoutFC);

            Sequence indicesSequence = new Sequence(trainingSet.Size);
            int[] miniBatch = new int[miniBatchSize];

            NetworkEvaluator networkEvaluator = new NetworkEvaluator();

            // Timing
            Stopwatch stopwatch = Stopwatch.StartNew();
            Stopwatch stopwatchFwd = Stopwatch.StartNew();
            Stopwatch stopwatchGrad = Stopwatch.StartNew();
            Stopwatch stopwatchBwd = Stopwatch.StartNew();

            int epoch = 0;
            int epochsRemainingToOutput = (evaluateBeforeTraining == true) ? 0 : consoleOutputLag;
            bool stopFlag = false;

            //using (System.IO.StreamWriter iterOutputFile = new System.IO.StreamWriter(trainingIterSavePath, true))
            //{
            while (epoch < maxTrainingEpochs && !stopFlag) // loop over training epochs
            {
                // Epoch beginning

                if (epochsRemainingToOutput == 0)
                {
                    /**************
                        * Evaluation *
                        **************/

                    // Evaluate on training set...
                    Console.WriteLine("Evaluating on TRAINING set...");
                    stopwatch.Restart();
                    networkEvaluator.EvaluateNetwork(network, trainingSet, out lossTraining, out errorTraining);
                    Console.WriteLine("\tLoss = {0}\n\tError = {1}\n\tEval runtime = {2}ms\n",
                                        lossTraining, errorTraining, stopwatch.ElapsedMilliseconds);
                    // ...and save to file
                    using (System.IO.StreamWriter trainingEpochOutputFile = new System.IO.StreamWriter(trainingEpochSavePath, true))
                    {
                        trainingEpochOutputFile.WriteLine(lossTraining.ToString() + "\t" + errorTraining.ToString());
                    }

                    // Evaluate on validation set...
                    if (validationSet != null)
                    {
                        double tmpErrorValidation;
                        Console.WriteLine("Evaluating on VALIDATION set...");
                        stopwatch.Restart();
                        networkEvaluator.EvaluateNetwork(network, validationSet, out lossValidation, out tmpErrorValidation);
                        Console.WriteLine("\tLoss = {0}\n\tError = {1}\n\tEval runtime = {2}ms\n",
                                            lossValidation, tmpErrorValidation, stopwatch.ElapsedMilliseconds);
                        // ...and save to file
                        using (System.IO.StreamWriter validationEpochOutputFile = new System.IO.StreamWriter(validationEpochSavePath, true))
                        {
                            validationEpochOutputFile.WriteLine(lossValidation.ToString() + "\t" + tmpErrorValidation.ToString());
                        }

                        if (tmpErrorValidation < errorValidation || !earlyStopping)
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
                }
                epochsRemainingToOutput--;

                /************
                    * Training *
                    ************/

                network.Set("Training", true);
                network.Set("EpochBeginning", true);

                Console.Write("\nEpoch {0}...", epoch);

                // Dropout
                if (epoch < epochsBeforeDropout)
                {
                    network.Set("DropoutConv", 1.0);
                    network.Set("DropoutFC", 1.0);
                }
                else if (epoch == epochsBeforeDropout)
                {
                    network.Set("DropoutConv", dropoutConv);
                    network.Set("DropoutFC", dropoutFC);
                }

                stopwatch.Restart();
                stopwatchFwd.Reset();
                stopwatchGrad.Reset();
                stopwatchBwd.Reset();

                indicesSequence.Shuffle(); // shuffle examples order at every epoch

                int iMiniBatch = 0;
                // Run over mini-batches 
                for (int iStartMiniBatch = 0; iStartMiniBatch < trainingSet.Size; iStartMiniBatch += miniBatchSize)
                {
                    // Feed a mini-batch to the network
                    miniBatch = indicesSequence.GetMiniBatchIndices(iStartMiniBatch, miniBatchSize);
                    network.InputLayer.FeedData(trainingSet, miniBatch);

                    // Forward pass
                    stopwatchFwd.Start();
                    network.ForwardPass();
                    stopwatchFwd.Stop();

                    /*
                    if (saveIters)
                    {
                        // Compute loss and error on this mini-batch and save to text file
                        double lossBatch;
                        double errorBatch;
                        networkEvaluator.ComputeBatchLossError(network, trainingSet, miniBatch, out lossBatch, out errorBatch);
                        iterOutputFile.WriteLine(lossBatch.ToString() + "\t" + errorBatch.ToString());
                    }
                    */

                    // Compute gradient and backpropagate 
                    stopwatchGrad.Start();
                    network.CrossEntropyGradient(trainingSet, miniBatch);
                    stopwatchGrad.Stop();

                    // Backpropagate gradient and update parameters
                    stopwatchBwd.Start();
                    network.BackwardPass(learningRate, momentumMultiplier, weightDecayCoeff);
                    stopwatchBwd.Stop();

                    iMiniBatch++;
                } // end of training epoch

                Console.Write(" Training runtime = {0}ms\n", stopwatch.ElapsedMilliseconds);

                Console.WriteLine("Forward: {0}ms - Gradient: {1}ms - Backward: {2}ms\n",
                    stopwatchFwd.ElapsedMilliseconds, stopwatchGrad.ElapsedMilliseconds, stopwatchBwd.ElapsedMilliseconds);

#if TIMING_LAYERS
                Console.WriteLine("\n DETAILED RUNTIMES:");

                Console.WriteLine("\n\tConvForward: {0}ms \n\tConvBackprop: {1}ms \n\tConvUpdateSpeeds: {2}ms \n\tConvUpdateParameters: {3}ms",
                    Utils.ConvForwardTimer.ElapsedMilliseconds, Utils.ConvBackpropTimer.ElapsedMilliseconds, 
                    Utils.ConvUpdateSpeedsTimer.ElapsedMilliseconds, Utils.ConvUpdateParametersTimer.ElapsedMilliseconds);

                Console.WriteLine("\n\tPoolingForward: {0}ms \n\tPoolingBackprop: {1}ms",
                    Utils.PoolingForwardTimer.ElapsedMilliseconds, Utils.PoolingBackpropTimer.ElapsedMilliseconds);

                Console.WriteLine("\n\tNonlinearityForward: {0}ms \n\tNonlinearityBackprop: {1}ms",
                    Utils.NonlinearityForwardTimer.ElapsedMilliseconds, Utils.NonlinearityBackpropTimer.ElapsedMilliseconds);

                Console.WriteLine("\n\tFCForward: {0}ms \n\tFCBackprop: {1}ms \n\tFCUpdateSpeeds: {2}ms \n\tFCUpdateParameters: {3}ms",
                        Utils.FCForwardTimer.ElapsedMilliseconds, Utils.FCBackpropTimer.ElapsedMilliseconds,
                        Utils.FCUpdateSpeedsTimer.ElapsedMilliseconds, Utils.FCUpdateParametersTimer.ElapsedMilliseconds);

                Console.WriteLine("\n\tSoftmax: {0}ms", Utils.SoftmaxTimer.ElapsedMilliseconds);

                Utils.ResetTimers();
#endif
                epoch++;

                if (Console.KeyAvailable)
                {
                    if (Console.ReadKey(true).Key == ConsoleKey.S)
                    {
                        Console.WriteLine("Key 'S' pressed! Stopping training...");
                        stopFlag = true;
                    }
                    else if (Console.ReadKey(true).Key == ConsoleKey.L)
                    {
                        learningRate /= 10;
                        Console.WriteLine("Key 'L' pressed! \n\tReducing learning rate by a factor of 10.\n\tWas{0}, now is {1}", 10 * learningRate, learningRate);
                    }
                    else
                        Console.WriteLine("That key has no effect... Press 'S' to stop training.");
                }

            }

            stopwatch.Stop();

        //}
        }

        #region Deprecated methods

        /*

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
