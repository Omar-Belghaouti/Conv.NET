using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{ 
    static class NetworkTrainer
    {
        #region Fields

        // Hyperparameters
        private static double learningRate;
        private static double momentumCoefficient;
        private static double weightDecayCoeff;
        private static int maxTrainingEpochs;
        private static int miniBatchSize;
        private static double errorTolerance;
        private static int consoleOutputLag;
        private static bool evaluateBeforeTraining;
        private static bool earlyStopping;
        private static double dropoutFC;
        private static double dropoutConv;
        private static double dropoutInput;
        private static int epochsBeforeRegularization;
        private static int patience;
        private static double learningRateDecayFactor;
        private static int maxConsecutiveAnnealings;
        private static double weightMaxNorm;

        // Losses/Errors
        private static double lossTraining;
        private static double minLossValidation = double.PositiveInfinity;
        private static double newLossValidation;
        private static double errorTraining;
        private static double errorValidation;
        private static double newErrorValidation;
        
        // auxiliary fields
        private static string trainingMode;

        // Paths for saving data
        private static string trainingEpochSavePath;
        private static string validationEpochSavePath;
        private static string networkOutputFilePath;

        #endregion


        #region Properties

        public static double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }

        public static double LearningRateDecayFactor
        {
            get { return learningRateDecayFactor; }
            set { learningRateDecayFactor = value; }
        }

        public static double MomentumCoefficient
        {
            get { return momentumCoefficient; }
            set { momentumCoefficient = value; }
        }

        public static double WeightDecayCoeff
        {
            get { return weightDecayCoeff; }
            set { weightDecayCoeff = value; }
        }

        public static int MaxTrainingEpochs
        {
            get { return maxTrainingEpochs; }
            set { maxTrainingEpochs = value; }
        }

        public static int EpochsBeforeRegularization
        {
            get { return epochsBeforeRegularization; }
            set { epochsBeforeRegularization = value; }
        }

        public static int MiniBatchSize
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

        public static double LossTraining
        {
            get { return lossTraining; }
        }

        public static double LossValidation
        {
            get { return minLossValidation; }
        }

        public static double ErrorTraining
        {
            get { return errorTraining; }
        }

        public static double ErrorValidation
        {
            get { return errorValidation; }
        }

        public static bool EvaluateBeforeTraining
        {
            set { evaluateBeforeTraining = value; }
        }

        public static bool EarlyStopping
        {
            set { earlyStopping = value; }
        }

        public static double WeightMaxNorm
        {
            get { return weightMaxNorm; }
            set { weightMaxNorm = value; }
        }

        public static double DropoutFullyConnected
        {
            get { return dropoutFC; }
            set { dropoutFC = value; }
        }

        public static double DropoutConvolutional
        {
            get { return dropoutConv; }
            set { dropoutConv = value; }
        }

        public static double DropoutInput
        {
            get { return dropoutInput; }
            set { dropoutInput = value; }
        }

        public static string TrainingEpochSavePath
        {
            set { trainingEpochSavePath = value; }
        }

        public static string ValidationEpochSavePath
        {
            set { validationEpochSavePath = value; }
        }

        public static string NetworkOutputFilePath
        {
            set { networkOutputFilePath = value; }
        }

        public static int Patience
        {
            set { patience = value; }
        }

        public static int MaxConsecutiveAnnealings
        {
            set { maxConsecutiveAnnealings = value; }
        }
        
        public static string TrainingMode
        {
            set { trainingMode = value; }
        }

        #endregion


        public static void Train(NeuralNetwork network, DataSet trainingSet, DataSet validationSet)
        {
            

            // Initialize parameters or load them
            if (trainingMode == "new" || trainingMode == "New")
            {
                // Setup miniBatchSize
                network.Set("MiniBatchSize", miniBatchSize);
                network.InitializeParameters("random");
            }
            else if (trainingMode == "resume" || trainingMode == "Resume")
                network.InitializeParameters("load");
            else
                throw new InvalidOperationException("Please set TrainingMode to either ''New'' or ''Resume''.");
            
            // Set dropout
            network.Set("DropoutFC", dropoutFC);
            network.Set("DropoutConv", dropoutConv);
            network.Set("DropoutInput", dropoutInput);

            Sequence indicesSequence = new Sequence(trainingSet.Size);
            int[] miniBatch = new int[miniBatchSize];

            // Timers
            Stopwatch stopwatch = Stopwatch.StartNew();
            Stopwatch stopwatchFwd = Stopwatch.StartNew();
            Stopwatch stopwatchGrad = Stopwatch.StartNew();
            Stopwatch stopwatchBwd = Stopwatch.StartNew();

            int epoch = 0;
            int nBadEpochs = 0;
            int consecutiveAnnealingCounter = 0;
            bool stopFlag = false;
            int epochsRemainingToOutput = (evaluateBeforeTraining == true) ? 0 : consoleOutputLag;
            

            while (!stopFlag) // begin loop over training epochs
            {
                
                if (epochsRemainingToOutput == 0)
                {
                    /**************
                     * Evaluation *
                     **************/

                    // Pre inference (for batch-norm)
                    //network.Set("PreInference", true);
                    //Console.WriteLine("Re-computing batch-norm means and variances...");
                    //NetworkEvaluator.PreEvaluateNetwork(network, trainingSet);

                    // Evaluate on training set...
                    network.Set("Inference", true);
                    Console.WriteLine("Evaluating on TRAINING set...");
                    stopwatch.Restart();
                    NetworkEvaluator.EvaluateNetwork(network, trainingSet, out lossTraining, out errorTraining);
                    Console.WriteLine("\tLoss = {0}\n\tError = {1}\n\tEval runtime = {2}ms\n",
                                        lossTraining, errorTraining, stopwatch.ElapsedMilliseconds);
                    // ...and save loss and error to file
                    using (System.IO.StreamWriter trainingEpochOutputFile = new System.IO.StreamWriter(trainingEpochSavePath, true))
                    {
                        trainingEpochOutputFile.WriteLine(lossTraining.ToString() + "\t" + errorTraining.ToString());
                    }

                    // Evaluate on validation set...
                    if (validationSet != null)
                    {
                        Console.WriteLine("Evaluating on VALIDATION set...");
                        stopwatch.Restart();
                        NetworkEvaluator.EvaluateNetwork(network, validationSet, out newLossValidation, out newErrorValidation);
                        Console.WriteLine("\tLoss = {0}\n\tError = {1}\n\tEval runtime = {2}ms\n",
                                            newLossValidation, newErrorValidation, stopwatch.ElapsedMilliseconds);
                        // ...save loss and error to file
                        using (System.IO.StreamWriter validationEpochOutputFile = new System.IO.StreamWriter(validationEpochSavePath, true))
                        {
                            validationEpochOutputFile.WriteLine(newLossValidation.ToString() + "\t" + newErrorValidation.ToString());
                        }

                        if (newLossValidation < minLossValidation)
                        {
                            // nice, validation loss is decreasing!
                            minLossValidation = newLossValidation;
                            errorValidation = newErrorValidation;

                            // Save network to file
                            Utils.SaveNetworkToFile(network, networkOutputFilePath);

                            // and keep training
                            nBadEpochs = 0;
                            consecutiveAnnealingCounter = 0;
                        }
                        else
                        {
                            nBadEpochs++;
                            Console.WriteLine("Loss on the validation set has been increasing for {0} epoch(s)...", nBadEpochs);
                            if (patience - nBadEpochs > 0)
                                Console.WriteLine("...I'll be patient for {0} more epoch(s)!", patience - nBadEpochs); // keep training
                            else
                            {
                                //Console.WriteLine("...and I've run out of patience! Training ends here.");
                                //stopFlag = true;
                                //break;

                                // Decrease learning rate
                                Console.WriteLine("...and I've run out of patience!");

                                if (consecutiveAnnealingCounter  > maxConsecutiveAnnealings)
                                {
                                    Console.WriteLine("\nReached the numner of maximum consecutive annealings without progress. \nTraining ends here.");
                                    break;        
                                }

                                Console.WriteLine("\nI'm annealing the learning rate:\n\tWas {0}\n\tSetting it to {1}.", learningRate, learningRate/learningRateDecayFactor);
                                learningRate /= learningRateDecayFactor;
                                consecutiveAnnealingCounter++;

                                Console.WriteLine("\nAnd I'm loading the network saved {0} epochs ago and resume the training from there.", patience);

                                string networkName = network.Name;
                                network = null; // this is BAD PRACTICE
                                GC.Collect(); // this is BAD PRACTICE
                                network = Utils.LoadNetworkFromFile("../../../../Results/Networks/", networkName);
                                network.Set("MiniBatchSize", miniBatchSize);
                                network.InitializeParameters("load");


                                nBadEpochs = 0;
                            }
                        }
                    }

                    // Restore dropout
                    network.Set("DropoutFC", dropoutFC);
                    network.Set("DropoutConv", dropoutConv);
                    network.Set("DropoutInput", dropoutInput);

                    epochsRemainingToOutput = consoleOutputLag;
                }
                epochsRemainingToOutput--;

                epoch++;

                if (epoch > maxTrainingEpochs)
                    break;

                /************
                * Training *
                ************/

                network.Set("Training", true);
                network.Set("EpochBeginning", true);

                Console.WriteLine("\nEpoch {0}...", epoch);


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
                    network.ForwardPass("beginning", "end");
                    stopwatchFwd.Stop();

                    // Compute gradient and backpropagate 
                    stopwatchGrad.Start();
                    network.CrossEntropyGradient(trainingSet, miniBatch);
                    stopwatchGrad.Stop();

                    // Backpropagate gradient and update parameters
                    stopwatchBwd.Start();
                    network.BackwardPass(learningRate, momentumCoefficient, weightDecayCoeff, weightMaxNorm);
                    stopwatchBwd.Stop();

                    iMiniBatch++;

                    CheckForKeyPress(ref network, ref stopFlag);
                    if (stopFlag)
                        break;

                } // end of training epoch

                Console.Write(" Training runtime = {0}ms\n", stopwatch.ElapsedMilliseconds);

                Console.WriteLine("Forward: {0}ms - Gradient: {1}ms - Backward: {2}ms\n",
                    stopwatchFwd.ElapsedMilliseconds, stopwatchGrad.ElapsedMilliseconds, stopwatchBwd.ElapsedMilliseconds);

#if TIMING_LAYERS
                Console.WriteLine("\n Detailed runtimes::");

                Console.WriteLine("\nCONV: \n\tForward: {0}ms \n\tBackprop: {1}ms \n\tUpdateSpeeds: {2}ms \n\tUpdateParameters: {3}ms \n\tPadUnpad: {4}ms",
                    Utils.ConvForwardTimer.ElapsedMilliseconds, Utils.ConvBackpropTimer.ElapsedMilliseconds, 
                    Utils.ConvUpdateSpeedsTimer.ElapsedMilliseconds, Utils.ConvUpdateParametersTimer.ElapsedMilliseconds, Utils.ConvPadUnpadTimer.ElapsedMilliseconds);

                Console.WriteLine("\nPOOLING: \n\tForward: {0}ms \n\tBackprop: {1}ms",
                    Utils.PoolingForwardTimer.ElapsedMilliseconds, Utils.PoolingBackpropTimer.ElapsedMilliseconds);

                Console.WriteLine("\nNONLINEARITIES: \n\tForward: {0}ms \n\tBackprop: {1}ms",
                    Utils.NonlinearityForwardTimer.ElapsedMilliseconds, Utils.NonlinearityBackpropTimer.ElapsedMilliseconds);

                Console.WriteLine("\nFULLY CONNECTED: \n\tForward: {0}ms \n\tBackprop: {1}ms \n\tUpdateSpeeds: {2}ms \n\tUpdateParameters: {3}ms",
                        Utils.FCForwardTimer.ElapsedMilliseconds, Utils.FCBackpropTimer.ElapsedMilliseconds,
                        Utils.FCUpdateSpeedsTimer.ElapsedMilliseconds, Utils.FCUpdateParametersTimer.ElapsedMilliseconds);

                Console.WriteLine("\nBATCHNORM FC \n\tForward: {0}ms \n\tBackprop: {1}ms \n\tUpdateSpeeds: {2}ms \n\tUpdateParameters: {3}ms",
                        Utils.BNFCForwardTimer.ElapsedMilliseconds, Utils.BNFCBackpropTimer.ElapsedMilliseconds,
                        Utils.BNFCUpdateSpeedsTimer.ElapsedMilliseconds, Utils.BNFCUpdateParametersTimer.ElapsedMilliseconds);

                 Console.WriteLine("\nBATCHNORM CONV \n\tForward: {0}ms \n\tBackprop: {1}ms \n\tUpdateSpeeds: {2}ms \n\tUpdateParameters: {3}ms",
                        Utils.BNConvForwardTimer.ElapsedMilliseconds, Utils.BNConvBackpropTimer.ElapsedMilliseconds,
                        Utils.BNConvUpdateSpeedsTimer.ElapsedMilliseconds, Utils.BNConvUpdateParametersTimer.ElapsedMilliseconds);

                Console.WriteLine("\nSOFTMAX \n\tForward: {0}ms", Utils.SoftmaxTimer.ElapsedMilliseconds);

                Utils.ResetTimers();
#endif
                

                

            }

            stopwatch.Stop();

        
        }


        private static void CheckForKeyPress(ref NeuralNetwork network, ref bool stopFlag)
        {
            ConsoleKeyInfo key = new ConsoleKeyInfo();

            if (Console.KeyAvailable)
            {
                key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.Escape)
                {
                    Console.WriteLine("\nEscape key pressed! Press it again to STOP training. Press any other key to resume.");
                    ConsoleKeyInfo oneMoreKey = Console.ReadKey();
                    if (oneMoreKey.Key == ConsoleKey.Escape)
                    {
                        Console.WriteLine("You pressed Escape again. Training will stop.");
                        stopFlag = true;
                    }
                }
                else if (key.Key == ConsoleKey.L)
                {
                    Console.Write("\nKey 'L' pressed! \n\tCurrent learning rate: {0}\n\tEnter new learning rate: ", learningRate);
                    string userInput = Console.ReadLine();
                    double newLearningRate;
                    if (double.TryParse(userInput, out newLearningRate))
                    {
                        learningRate = newLearningRate;
                        Console.WriteLine("\n\nLearning rate is now {0}", learningRate);
                    }
                    else
                    {
                        Console.WriteLine("Not a valid learning rate.");
                    }
                }
                else if (key.Key == ConsoleKey.W)
                {
                    Console.Write("\nKey 'W' pressed! \n\tCurrent weight decay coefficient: {0}\n\tEnter new weight decay coefficient: ", weightDecayCoeff);
                    string userInput = Console.ReadLine();
                    double newWeightDecayCoeff;
                    if (double.TryParse(userInput, out newWeightDecayCoeff))
                    {
                        weightDecayCoeff = newWeightDecayCoeff;
                        Console.WriteLine("\n\nWeight decay coefficient is now {0}", weightDecayCoeff);
                    }
                    else
                    {
                        Console.WriteLine("Not a valid weight decay coefficient.");
                    }
                }
                else if (key.Key == ConsoleKey.S)
                {
                    Console.Write("\nKey 'S' pressed! \n\tSaving the current state of the network to file.\n\tEnter file path: ");
                    string outputFilePath = Console.ReadLine();
                    Utils.SaveNetworkToFile(network, outputFilePath);
                }
                else
                    Console.WriteLine("That key has no effect...\n\tPress Escape to stop training.\n\tPress 'L' to change the learning rate.\n\tPress 'W' to change the weight decay coefficient.\n\tPress 'S' to save the current state of the network to file.");
            }

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
