using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;
using System.Diagnostics;

namespace trafficNetCL
{
    class NeuralNetwork
    {
        // Private fields
        public List<Layer> Layers; // Layers of the network
        DataTable TrainingSet; // setting datasets as class fields could be pointless...think about it
        DataTable ValidationSet;
        DataTable TestSet;

        public double trainingError;

        // Constructor
        public NeuralNetwork()
        {
            Console.WriteLine("--- New empty network created ---");
            this.Layers = new List<Layer>(); // empty list of Layers
        }

        // Add Layers to the network, in a linked fashion (...really needed??)
        public void AddLayer(Layer layer)
        {
            bool isNotEmpty = Layers.Any();
            if (isNotEmpty)
            {
                Layers.Last().NextLayer = layer;
            }
            Layers.Add(layer);
        }

        /// <summary>
        /// Setup network: given input dim and each layer's parameters, automatically set dimensions of I/O 3D arrays and initialize weights and biases.
        /// </summary>
        /// <param name="inputDimensions"></param>
        /// <param name="nOutputClasses"></param>
        public void Setup(int[] inputDimensions, int nOutputClasses)
        {
            Console.WriteLine("--- Network setup started ---");

            Console.WriteLine("Setting up layer 0...");
            Layers[0].SetupInput(inputDimensions[0], inputDimensions[1], inputDimensions[2]);
            Layers[0].SetupOutput();
            Layers[0].InitializeWeightsAndBiases();

            for (int i = 1; i < Layers.Count; i++ ) // all other layers
            {
                Console.WriteLine("Setting up layer {0}...", i);
                Layers[i].SetupInput(Layers[i-1].OutputWidth, Layers[i-1].OutputHeight, Layers[i-1].OutputDepth);
                Layers[i].SetupOutput();
                Layers[i].InitializeWeightsAndBiases();
            }

            Console.WriteLine("--- Network setup complete ---");
        }





        
        void train( DataTable trainingSet, 
                    DataTable validationSet, 
                    double learningRate, 
                    double momentumMultiplier, 
                    int maxTrainingEpochs, 
                    int miniBatchSize,
                    double errorTolerance) // returns training error
        {   
            this.TrainingSet = trainingSet;
            int sizeTrainingSet = trainingSet.Rows.Count;
            Debug.Assert(sizeTrainingSet % miniBatchSize == 0);
            int nMiniBatches = sizeTrainingSet / miniBatchSize;

            this.ValidationSet = validationSet;

            bool stopFlag = false;
            int epoch = 0;
            while (epoch < maxTrainingEpochs && !stopFlag) 
            {
         
                // TO-DO: split training set into mini-batches

                // TO-DO: implement training
                // At the end of the epoch we should get a training error and a validation error


                if (trainingError < errorTolerance)
                    stopFlag = true;
                // TO-DO: also implement early stopping (stop if validation error starts increasing)
         
            }
        }

        double test(DataTable testSet)
        {
            int nCorrectClassifications = 0;
            float[] outputScores;
            int assignedClass;

            // TO-DO: transform this to parallelized GPU code
            foreach (DataRow dataRow in testSet.Rows)
            {
                float[, ,] inputImage = dataRow.Field<float[, ,]>(0);
                int targetClass = dataRow.Field<int>(1);

                Layers[0].Input = inputImage;
                for (int i = 0; i < Layers.Count; i++)
                {
                    Layers[i].ForwardOne(); // run layer forward
                    if (i < Layers.Count - 1) // if it is not the last layer
                        Layers[i + 1].Input = Layers[i].Output; // then set output as input of next layer
                }

                outputScores = Layers[Layers.Count - 1].Output.Cast<float>().ToArray(); // cast output of last layer to 1D array
                assignedClass = indexMaxScore(outputScores);

                if (assignedClass == targetClass)
                    nCorrectClassifications += 1;
            }

            return (double)nCorrectClassifications / (double)testSet.Rows.Count;
        }

        
		int indexMaxScore(float[] outputScores){
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
