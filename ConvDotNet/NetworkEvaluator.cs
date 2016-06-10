using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    public static class NetworkEvaluator
    {


        /// <summary>
        /// Run this method before evaluation, passing the TRAINING set as second argument.
        /// This will compute cumulative averages needed for inference in BatchNormConv layers, if any.
        /// </summary>
        /// <param name="network"></param>
        /// <param name="dataSet"></param>
        public static void PreEvaluateNetwork(NeuralNetwork network, DataSet dataSet)
        {
            // Set network for pre-inference (needed for BatchNorm layers)
            network.Set("PreInference", true);

            // Turn off dropout
            network.Set("DropoutFC", 1.0);
            network.Set("DropoutConv", 1.0);
            network.Set("DropoutInput", 1.0);

            int miniBatchSize = network.Layers[0].OutputNeurons.MiniBatchSize;
            
            Sequence indicesSequence = new Sequence(dataSet.Size);

            // Run over mini-batches (in order, no shuffling)
            for (int iStartMiniBatch = 0; iStartMiniBatch < dataSet.Size; iStartMiniBatch += miniBatchSize)  
            {
                // Feed a mini-batch to the network
                int[] miniBatch = indicesSequence.GetMiniBatchIndices(iStartMiniBatch, miniBatchSize);
                network.InputLayer.FeedData(dataSet, miniBatch);

                // Run network forward
                network.ForwardPass("beginning", "end");

                // Do not compute loss or error
                
            }
        }

        public static void EvaluateNetwork(NeuralNetwork network, DataSet dataSet, out double loss, out double error)
        {
            // Set network for inference (needed for BatchNorm layers)
            network.Set("Inference", true);

            loss = 0.0;
            error = 0.0;

            // Turn off dropout
            network.Set("DropoutFC", 1.0);
            network.Set("DropoutConv", 1.0);
            network.Set("DropoutInput", 1.0);

            int miniBatchSize = network.Layers[0].OutputNeurons.MiniBatchSize;
            
            Sequence indicesSequence = new Sequence(dataSet.Size);

            // Run over mini-batches (in order, no shuffling here)
            for (int iStartMiniBatch = 0; iStartMiniBatch < dataSet.Size; iStartMiniBatch += miniBatchSize)  
            {
                // Feed a mini-batch to the network
                int[] miniBatch = indicesSequence.GetMiniBatchIndices(iStartMiniBatch, miniBatchSize);
                network.InputLayer.FeedData(dataSet, miniBatch);

                // Run network forward
                network.ForwardPass("beginning", "end");


                for (int m = 0; m < Math.Min(miniBatchSize,dataSet.Size-iStartMiniBatch) ; m++) // In case dataSet.Size doesn't divide miniBatchSize, the last miniBatch contains copies! Don't want to re-evaluate them
                {
                    double[] outputScores = network.OutputLayer.OutputClassScores[m];

                    int assignedLabel = Utils.IndexOfMax(outputScores);
                    int trueLabel = dataSet.Labels[miniBatch[m]];

                    // Cumulate loss and error
                    loss -= Math.Log(outputScores[trueLabel]);
                    error += (assignedLabel == trueLabel) ? 0 : 1;

                } // end loop within a mini-batch
                
            } // end loop over mini-batches
             
            error /= dataSet.Size;
            loss /= dataSet.Size;
        }


        public static void ComputeBatchLossError(NeuralNetwork network, DataSet dataSet, int[] miniBatch, out double loss, out double error)
        {
            loss = 0.0;
            error = 0.0;

            // Find maximum output score (i.e. assigned class) of each mini batch item
            for (int m = 0; m < miniBatch.Length; m++)
            {
                double[] outputScores = network.OutputLayer.OutputClassScores[m];

                int assignedLabel = Utils.IndexOfMax(outputScores);
                int trueLabel = dataSet.Labels[miniBatch[m]];

                // Cumulate loss and error
                loss -= Math.Log(outputScores[trueLabel]);
                error += (assignedLabel == trueLabel) ? 0 : 1;

            } // end loop within a mini-batch

            error /= miniBatch.Length;
            loss /= miniBatch.Length;
        }




        /// <summary>
        /// Ensemble the predictions of a list of neural networks by averaging them. TODO: also try voting.
        /// </summary>
        /// <param name="NetworkEnsemble"></param>
        /// <param name="GrayscaleDataset"></param>
        /// <param name="RGBDataset"></param>
        /// <param name="miniBatchSize"></param>
        /// <param name="loss"></param>
        /// <param name="error"></param>
        public static void EvaluateEnsemble(List<NeuralNetwork> NetworkEnsemble, DataSet GrayscaleDataset, DataSet RGBDataset, int miniBatchSize, out double error)
        {
            // This is a TERRIBLE piece of code. I am myself disgusted by what I've written below. But I've slept ~20 hours in the past 5 days and I'm tired as fuck.

            error = 0.0;

            int nModels = NetworkEnsemble.Count;
            int nExamples = GrayscaleDataset.Size;
            int nClasses = GrayscaleDataset.NumberOfClasses;

            // Prepare networks for evaluation
            for (int i = 0; i < nModels; ++i)
            {
                // Set mini-batch size
                NetworkEnsemble[i].Set("MiniBatchSize", miniBatchSize); // the value here doesn't matter,it's just for computational efficiency

                // Load network's parameters
                NetworkEnsemble[i].InitializeParameters("load");

                // Set network for inference (needed for BatchNorm layers)
                NetworkEnsemble[i].Set("Inference", true);

                // Turn off dropout
                NetworkEnsemble[i].Set("DropoutFC", 1.0);
                NetworkEnsemble[i].Set("DropoutConv", 1.0);
                NetworkEnsemble[i].Set("DropoutInput", 1.0);
            }

            Sequence indicesSequence = new Sequence(nExamples);

            // Run over mini-batches (in order, no shuffling here)
            for (int iStartMiniBatch = 0; iStartMiniBatch < nExamples; iStartMiniBatch += miniBatchSize)
            {
                List<double[]> miniBatchClassScores = new List<double[]>();
                for (int m = 0; m < Math.Min(miniBatchSize, nExamples - iStartMiniBatch); m++) // In case dataSet.Size doesn't divide miniBatchSize, the last miniBatch contains copies! Don't want to re-evaluate them
                    miniBatchClassScores.Add(new double[nClasses]);

                // Feed this mini-batch to the network
                int[] miniBatch = indicesSequence.GetMiniBatchIndices(iStartMiniBatch, miniBatchSize);

                foreach (NeuralNetwork network in NetworkEnsemble)
                {
                    if (network.InputLayer.OutputDepth == 1)
                        network.InputLayer.FeedData(GrayscaleDataset, miniBatch);
                    else if (network.InputLayer.OutputDepth == 3)
                        network.InputLayer.FeedData(RGBDataset, miniBatch);
                    else
                        throw new InvalidOperationException("Input layer property OutputDepth does not return 1 nor 3.");

                    // Run network forward
                    network.ForwardPass("beginning", "end");

                    for (int m = 0; m < Math.Min(miniBatchSize, nExamples - iStartMiniBatch); m++) // In case dataSet.Size doesn't divide miniBatchSize, the last miniBatch contains copies! Don't want to re-evaluate them
                    {
                        double[] thisOutputScores = network.OutputLayer.OutputClassScores[m];

                        // cumulate the outputScores array of this netowrk on the m-th entry of the list
                        miniBatchClassScores[m] = Utils.AddArrays(miniBatchClassScores[m], thisOutputScores); 
                    }
                } // end loop over networks

                // Now miniBatchClassScores contains M arrays, each of which represent an example within this mini-batch 
                // and contains the cumulated class scores from all networks. Now just take the max.

                for (int m = 0; m < Math.Min(miniBatchSize, nExamples - iStartMiniBatch); m++) // In case dataSet.Size doesn't divide miniBatchSize, the last miniBatch contains copies! Don't want to re-evaluate them
                {
                    int assignedLabel = Utils.IndexOfMax(miniBatchClassScores[m]);
                    int trueLabel = GrayscaleDataset.Labels[miniBatch[m]];

                    // Cumulate loss and error
                    error += (assignedLabel == trueLabel) ? 0 : 1;
                } 
            } // end loop over mini-batches

            error /= nExamples;
        }




        public static void SaveMisclassifiedExamples(NeuralNetwork network, DataSet dataSet, string outputFilePath)
        {
            List<int> misclassifiedExamplesList = new List<int>();
            List<int> wrongLabels = new List<int>();

            // Set network for inference (needed for BatchNorm layers)
            network.Set("Inference", true);

            // Turn off dropout
            network.Set("DropoutFC", 1.0);
            network.Set("DropoutConv", 1.0);
            network.Set("DropoutInput", 1.0);

            int miniBatchSize = network.Layers[0].OutputNeurons.MiniBatchSize;

            Sequence indicesSequence = new Sequence(dataSet.Size);

            // Run over mini-batches (in order, no shuffling here)
            for (int iStartMiniBatch = 0; iStartMiniBatch < dataSet.Size; iStartMiniBatch += miniBatchSize)
            {
                // Feed a mini-batch to the network
                int[] miniBatch = indicesSequence.GetMiniBatchIndices(iStartMiniBatch, miniBatchSize);
                network.InputLayer.FeedData(dataSet, miniBatch);

                // Run network forward
                network.ForwardPass("beginning", "end");

                for (int m = 0; m < Math.Min(miniBatchSize, dataSet.Size - iStartMiniBatch); m++) // In case dataSet.Size doesn't divide miniBatchSize, the last miniBatch contains copies! Don't want to re-evaluate them
                {
                    double[] outputScores = network.OutputLayer.OutputClassScores[m];

                    int assignedLabel = Utils.IndexOfMax(outputScores);
                    int trueLabel = dataSet.Labels[miniBatch[m]];

                    if (assignedLabel != trueLabel)
                    {
                        misclassifiedExamplesList.Add(miniBatch[m]);
                        wrongLabels.Add(assignedLabel);
                    }
                } // end loop within a mini-batch

            } // end loop over mini-batches

            // Save the list to file
            using (System.IO.StreamWriter outputFile = new System.IO.StreamWriter(outputFilePath))
            {
                for (int i = 0; i < misclassifiedExamplesList.Count; ++i)
                {
                    outputFile.WriteLine(misclassifiedExamplesList[i].ToString() + "\t" + wrongLabels[i].ToString());
                }
                Console.WriteLine("Misclassified examples saved in file " + outputFilePath);
            }

        }


    }
}
