using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;

namespace TrafficNetCL
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
        public static int TrainSimpleTest(NeuralNetwork network, DataSet trainingSet, out double finalError, out int finalEpoch)
        {

            // Initializations
            int errorCode = 0;
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

                        network.Layers[0].Input.Set(trainingSet.GetDataPoint(iDataPoint));
                        // Run forward
                        for (int l = 0; l < network.NumberOfLayers; l++)
                        {
                            network.Layers[l].ForwardOneCPU();
                        }
                        outputScores = network.Layers.Last().Output.Get();
                        labelArray = trainingSet.GetLabelArray(iDataPoint);

                        // Gradient of quadratic cost, using LINQ
                        // network.Layers.Last().Output.Delta = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();
                        // network.Layers.Last().BackPropOneCPU();

                        // Gradient of cross-entropy cost (directly write in INPUT delta)
                        network.Layers.Last().Input.Delta = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();

                        // Now run backwards and update deltas (cumulating them), but DO NOT update parameters
                        for (int l = network.Layers.Count - 2; l >= 0; l--) // propagate deltas in all layers (but the last) backwards (L-2 to 0)
                        {
                            network.Layers[l].BackPropOneCPU();
                        }

                    } // end loop over mini-batches

                    // Now update parameters using cumulated deltas
                    for (int l = network.Layers.Count - 1; l >= 0; l--)
                    {
                        network.Layers[l].UpdateParameters(learningRate, momentumMultiplier);
                    }

                    // And finally wipe out all cumulated deltas (NOTE: can NOT merge this and previous loop!)
                    for (int l = network.Layers.Count - 1; l >= 0; l--)
                    {
                        network.Layers[l].ClearDelta();
                    }
                    

                }


                if (isOutputEpoch)
                {
                    //costEpoch = QuadraticCost(network, trainingSet);
                    errorEpoch = ClassificationErrorTopOne(network, trainingSet);
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


            finalEpoch = epoch;
            finalError = ClassificationErrorTopOne(network, trainingSet);

            return errorCode; // error code
        }


        /*
        static double QuadraticCost(float[] targetValues, float[] networkOutputs, out float[] gradient)
        {
            if (targetValues.Length != networkOutputs.Length)
                throw new System.InvalidOperationException("Mismatch between length of output array and target (label) array.");

            gradient = targetValues.Zip(networkOutputs, (x, y) => y - x).ToArray();
            var squaredErrors = gradient.Select(x => Math.Pow(x, 2));

            return squaredErrors.Sum() / squaredErrors.Count();
        }
         * 


        static double QuadraticCost(NeuralNetwork network, DataSet dataSet)
        {
            float[] dummy;
            double totalCost = 0;

            for (int i = 0; i < dataSet.Size; i++)
            {
                network.Layers[0].Input.Set(dataSet.GetDataPoint(i));

                // Run forward
                for (int l = 0; l < network.Layers.Count; l++)
                {
                    network.Layers[l].ForwardOneCPU();
                }

                // Compute cost
                totalCost += QuadraticCost(new float[] { (float)dataSet.GetLabel(i) }, network.Layers.Last().Output.Get(), out dummy);
            }

            return totalCost / (2 * dataSet.Size);
        }
         * */



        public static double TrainMNIST(NeuralNetwork network, DataSet trainingSet)
        {

            // Initializations
            int[] randomIntSequence = new int[trainingSet.Size];
            int iDataPoint;
            bool stopFlag = false;
            double errorEpoch;
            bool isOutputEpoch = true;
            int epochsRemainingToOutput = 0;
            
            float[] outputScores = new float[trainingSet.NumberOfClasses];
            float[] labelArray = new float[trainingSet.NumberOfClasses];

            int epoch = 0;

            do // loop over training epochs
            {
                randomIntSequence = Utils.GenerateRandomPermutation(trainingSet.Size);  // newly generated at every epoch

                // Run over mini-batches
                for (int iStartMiniBatch = 0; iStartMiniBatch < trainingSet.Size; iStartMiniBatch += miniBatchSize)
                {
                    // Run over a mini-batch
                    for (int iWithinMiniBatch = 0; iWithinMiniBatch < miniBatchSize; iWithinMiniBatch++)
                    {
                        iDataPoint = randomIntSequence[iStartMiniBatch + iWithinMiniBatch];

                        network.Layers[0].Input.Set(trainingSet.GetDataPoint(iDataPoint));
                        // Run forward
                        for (int l = 0; l < network.NumberOfLayers; l++)
                        {
                            network.Layers[l].ForwardOneCPU();
                        }
                        outputScores = network.Layers.Last().Output.Get();
                        labelArray = trainingSet.GetLabelArray(iDataPoint);

                        // JUST RUN FORWARD NOW

                        // Gradient of cross-entropy cost (directly write this to INPUT delta)
                        //network.Layers.Last().Input.Delta = outputScores.Zip(labelArray, (x, y) => (x - y)).ToArray();

                        // Now run backwards and update deltas (cumulating them), but DO NOT update parameters
                        /*
                        for (int l = network.Layers.Count - 2; l >= 0; l--) // propagate deltas in all layers (but the last) backwards (L-2 to 0)
                        {
                            network.Layers[l].BackPropOneCPU();
                        }
                        */

                    } // end loop over mini-batches

                    // Now update parameters using cumulated deltas
                    /*
                    for (int l = network.Layers.Count - 1; l >= 0; l--)
                    {
                        network.Layers[l].UpdateParameters(learningRate, momentumMultiplier);
                    }
                    

                    // And finally wipe out all cumulated deltas (NOTE: can NOT merge this and previous loop!)
                    for (int l = network.Layers.Count - 1; l >= 0; l--)
                    {
                        network.Layers[l].ClearDelta();
                    }
                    */

                }

                isOutputEpoch = epochsRemainingToOutput == 0;
                if (isOutputEpoch)
                {
                    //costEpoch = QuadraticCost(network, trainingSet);
                    errorEpoch = ClassificationErrorTopOne(network, trainingSet);
                    Console.WriteLine("Epoch {0}: classification error = {1}", epoch, errorEpoch);

                    if (errorEpoch < errorTolerance)
                        stopFlag = true;

                    epochsRemainingToOutput = consoleOutputLag;
                    isOutputEpoch = false;
                }
                epochsRemainingToOutput--;
                
                // TO-DO: also implement early stopping (stop if validation error starts increasing)
                epoch++;

            } while (epoch < maxTrainingEpochs && !stopFlag);

            return ClassificationErrorTopOne(network, trainingSet);
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
                network.Layers[0].Input.Set(dataSet.GetDataPoint(i));

                // Run forward
                for (int l = 0; l < network.Layers.Count; l++)
                {
                    network.Layers[l].ForwardOneCPU();
                }

                // Check for correct/wrong classification
                int outputClass = Math.Sign(network.Layers.Last().Output.Get()[0]);
                classificationError += Math.Abs(outputClass - dataSet.GetLabel(i));
            }

            return classificationError / (2* dataSet.Size);
        }



        static double ClassificationErrorTopOne(NeuralNetwork network, DataSet dataSet)
        {
            double classificationError = 0;

            for (int i = 0; i < dataSet.Size; i++)
            {
                network.Layers[0].Input.Set(dataSet.GetDataPoint(i));

                // Run forward
                for (int l = 0; l < network.Layers.Count; l++)
                {
                    network.Layers[l].ForwardOneCPU();
                }

                // Check for correct/wrong classification
                int outputClassMaxScore = IndexOfMax(network.Layers.Last().Output.Get());
                if (outputClassMaxScore != dataSet.GetLabel(i))
                {
                    classificationError += 1;
                }
            }

            return classificationError / dataSet.Size;
        }


        public static int IndexOfMax(float[] outputScores)
        {
            int iMax = 0;
            float max = outputScores[0];
            for (int j = 1; j < outputScores.Length; j++)
            {
                if (outputScores[j] > max)
                {
                    max = outputScores[j];
                    iMax = j;
                }
            }
            return iMax;
        }


        




    }
}
