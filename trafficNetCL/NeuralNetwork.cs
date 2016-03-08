using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;

namespace trafficNetCL
{
    class NeuralNetwork
    {
        // Members
        private List<Layer> layers; // Layers of the network
        private DataTable TrainingSet;
        private DataTable validationSet;
        private DataTable testSet;

        // Constructor
        public NeuralNetwork()
        {
            Console.WriteLine("Network created.");
            this.layers = new List<Layer>();
        }

        // Method to add layers
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
        }



        
        double train( DataTable trainingSet, int maxTrainingEpochs, int miniBatchSize ) // returns training error
        {   
            this.TrainingSet = trainingSet;
            int sizeTrainingSet = trainingSet.Rows.Count;
            int nMiniBatches = (int)Math.Floor((double)sizeTrainingSet / (double)miniBatchSize);  
         
            bool stopFlag = false;
            int epoch = 0;
         
            while (epoch < maxTrainingEpochs && !stopFlag) 
            {
         
                // split training set into mini-batches
          
         
         
            }

            return 0;
        }


    }
}
