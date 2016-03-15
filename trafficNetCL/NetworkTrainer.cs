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

            Debug.Assert(TrainingSet.Length % miniBatchSize == 0);
            int nMiniBatches = TrainingSet.Length / miniBatchSize;

            // TO-DO: split training set into mini-batches

            // TO-DO: implement single-epoch training

            // At the end of the epoch we should get a training error and a validation error
            //... compute online or "test" with whole training/validation sets??
            errorTraining = 1;
            errorValidation = 1;


            return errorCode;
        }

        public static int TrainSimpleTest(NeuralNetwork Network, float[] inputArray, float[] targetOutput, out double errorTraining)
        {
            int errorCode = 0;

            float[] costGgradient;

            bool stopFlag = false;
            int epoch = 0;
            do
            {
                Network.Layers[0].Input.Set(inputArray);

                // Run forward
                for (int l = 0; l < Network.Layers.Count; l++)
                {
                    Network.Layers[l].ForwardOneCPU();
                }

                // Compute cost
                errorTraining = QuadraticCost(targetOutput, Network.Layers.Last().Output.Get(), out costGgradient);
                Console.WriteLine("Iteration {1}: quadratic cost = {0}", errorTraining, epoch);

                // Error backpropagation and parameters update

                Network.Layers.Last().Output.Delta = costGgradient; // delta of output of last layer (L-1)

                for (int l = Network.Layers.Count-1; l >= 0; l--) // propagate deltas in all layers backwards (L-1 to 0)
                {
                    Network.Layers[l].BackPropOneCPU();
                    Network.Layers[l].UpdateParameters(learningRate);

                    /*
                    Console.WriteLine("\nInput delta of layer {0}:", l);
                    for (int i = 0; i < Network.Layers[l].Input.Delta.Length; i++)
                        Console.Write(Network.Layers[l].Input.Delta[i] + " ");
                    Console.WriteLine();
                    */
                }

                if (errorTraining < errorTolerance)
                    stopFlag = true;

                // TO-DO: also implement early stopping (stop if validation error starts increasing)
                epoch++;

            } while (epoch < maxTrainingEpochs && !stopFlag);

            return errorCode; // error code
        }




        static double QuadraticCost(float[] targetValues, float[] networkOutputs, out float[] gradient)
        {
            if (targetValues.Length != networkOutputs.Length)
                throw new System.InvalidOperationException("Mismatch between length of output array and target (label) array.");

            gradient = targetValues.Zip(networkOutputs, (x, y) => x - y).ToArray();
            var squaredErrors = gradient.Select(x => Math.Pow(x, 2));

            return squaredErrors.Sum() / squaredErrors.Count();
        }


    }
}
