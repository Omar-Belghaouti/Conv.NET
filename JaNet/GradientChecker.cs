using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JaNet
{
#if GRADIENT_CHECK
    static class GradientChecker
    {

        const double EPSILON = 0.0001;
        const double TOLERANCE_REL = 0.001;
        const double learningRate = 0;
        const double momentumMultiplier = 0;

        const int nPointsToCheck = 10;

        public static void Check(NeuralNetwork network, DataSet dataSet)
        {
            // for some random points in dataSet
            for (int iCheck = 0; iCheck < nPointsToCheck; iCheck++)
            {
                int iDataPoint = Global.rng.Next(0, dataSet.Size);

                // perform a SINGLE forward and backward pass with a SINGLE (random) example
                network.FeedDatum(dataSet, iDataPoint );
                network.ForwardPass();
                double[] outputScores = network.Layers.Last().OutputClassScores[0];
                int trueLabel = dataSet.GetLabel(iDataPoint);
                double loss = -Math.Log(outputScores[trueLabel]);
                network.CrossEntropyGradient(dataSet, new int[] { iDataPoint });
                network.BackwardPass(learningRate, momentumMultiplier);


                for (int iLayer = 0; iLayer < network.NumberOfLayers; iLayer++)
                {
                    Console.Write("\nChecking gradients in layer {0} ({1})...", iLayer, network.Layers[iLayer].Type);
                    

                    if (network.Layers[iLayer].Type != "Convolutional" && network.Layers[iLayer].Type != "FullyConnected")
                        Console.Write("OK (no parameters in this layer)");
                    else
                    {
                        int nErrors = 0;
                        int nChecks = 0;
                        // Get parameters
                        
                        //double[] biases = network.Layers[iLayer].Biases;

                        // Get gradients
                        
                        //double[] biasesGradients = network.Layers[iLayer].BiasesGradients;


                        for (int i = 0; i < network.Layers[iLayer].Weights.GetLength(0); i++)
                        {
                            for (int j = 0; j < network.Layers[iLayer].Weights.GetLength(1); j++)
                            {
                                double[,] weights = network.Layers[iLayer].Weights;
                                double[,] weightsGradients = network.Layers[iLayer].WeightsGradients;

                                if (Math.Abs(weightsGradients[i, j]) > EPSILON)
                                {
                                    nChecks++;
                                    //Console.Write("\nChecking gradient wrt weight {0}... ", i * weights.GetLength(1) + j);

                                    
                                    // decrease weight [i,j] by EPSILON and compute loss
                                    double[,] weightsMinus = weights;
                                    weightsMinus[i, j] -= EPSILON;
                                    network.Layers[iLayer].Weights = weightsMinus;
                                    network.FeedDatum(dataSet, iDataPoint );
                                    network.ForwardPass();
                                    double[] outputScoresPlus = network.Layers.Last().OutputClassScores[0];
                                    double lossMinus = -Math.Log(outputScoresPlus[trueLabel]);
                                    
                                    

                                    // increase weight [i,j] by EPSILON and compute loss
                                    double[,] weightsPlus = weights;
                                    weightsPlus[i, j] += EPSILON;
                                    network.Layers[iLayer].Weights = weightsPlus;
                                    network.FeedDatum(dataSet, iDataPoint );
                                    network.ForwardPass();
                                    double[] outputScoresMinus = network.Layers.Last().OutputClassScores[0];
                                    double lossPlus = -Math.Log(outputScoresMinus[trueLabel]);

                                    //double approxGradient = (lossPlus - loss) / (EPSILON);
                                    double approxGradient = (lossPlus - lossMinus) / (EPSILON);

                                    // compare gradient from backprop and approximate gradient
                                    double relativeError = Math.Abs(approxGradient - weightsGradients[i, j]) / 
                                        Math.Max( Math.Abs(weightsGradients[i, j]), Math.Abs(approxGradient) );
                                    if (relativeError > TOLERANCE_REL)
                                    {
                                        Console.Write("OUCH! Gradient check failed at weight {0}\n", i * weights.GetLength(1) + j);
                                        Console.WriteLine("\tBackpropagation gradient: {0}", weightsGradients[i, j]);
                                        Console.WriteLine("\tFinite difference gradient: {0}", approxGradient);
                                        Console.WriteLine("\tRelative error: {0}", relativeError);
                                        //Console.ReadKey();
                                        nErrors++;
                                    }

                                    

                                    // restore original weights before checking next
                                    network.Layers[iLayer].Weights = weights;
                                }
                            }
                        }
                        if (nErrors == 0)
                            Console.Write("OK");
                        if (nChecks == 0)
                            Console.Write("...maybe!");
                    }
                    

                }
            }


        }



    }
#endif
}
