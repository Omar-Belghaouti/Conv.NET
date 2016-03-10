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
        public static int Run(NeuralNetwork Network, DataSet TrainingSet, DataSet ValidationSet)
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

    }
}
