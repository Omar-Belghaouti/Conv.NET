using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace JaNet
{
    static class GradientChecker
    {
        const string typeToCheck = "BatchNormConv";

        const int miniBatchSize = 4;

        const double EPSILON = 0.00001;
        const double MAX_RELATIVE_ERROR = 0.01;


        public static void Check(NeuralNetwork network, DataSet dataSet)
        {
            // Setup network 

            network.Set("MiniBatchSize", miniBatchSize);
            network.InitializeParameters("random");
            network.Set("DropoutFC", 1.0);
            network.Set("Training", true);
            network.Set("EpochBeginning", true);

            // Get a mini-batch of data

            Sequence indicesSequence = new Sequence(dataSet.Size);
            indicesSequence.Shuffle();
            int[] miniBatch = indicesSequence.GetMiniBatchIndices(0, miniBatchSize);

            // Run network forward and backward

            network.InputLayer.FeedData(dataSet, miniBatch);
            network.ForwardPass("beginning", "end");
            List<int> trueLabels = new List<int>();
            for (int m = 0; m < miniBatchSize; m++)
                trueLabels.Add(dataSet.Labels[miniBatch[m]]);
            network.CrossEntropyGradient(dataSet, miniBatch);
            network.BackwardPass(0.0, 0.0, 0.0); // no momentum, no learning rate, no weight decay

            // Re-forward pass (in case there are batch-norm layer)
            network.Set("PreInference", true);
            network.ForwardPass("beginning", "end");
            network.Set("Inference", true);

            for (int iLayer = 1; iLayer < network.NumberOfLayers; iLayer++)
            {
                //if (network.Layers[iLayer].Type != "Input" && network.Layers[iLayer].Type != "MaxPooling" && network.Layers[iLayer].Type != "ReLU" &&
                //    network.Layers[iLayer].Type != "SoftMax" && network.Layers[iLayer].Type != "Convolutional" && network.Layers[iLayer].Type != "FullyConnected" 
                //    && network.Layers[iLayer].Type != "ELU")
                if (network.Layers[iLayer].Type == typeToCheck)
                {   
                    Console.WriteLine("\nChecking gradients in layer {0} ({1})...", iLayer, network.Layers[iLayer].Type);
                    int nChecks = 0;
                    int nErrors = 0;
                    double cumulativeError = 0.0;

                    double[] parametersBackup = network.Layers[iLayer].GetParameters();
                    double[] parameterGradients = network.Layers[iLayer].GetParameterGradients();
                    int nParameters = parametersBackup.Length;

                    // First parameters

                    Console.WriteLine("\n...with respect to PARAMETERS");
                    for (int j = 0; j < nParameters; j++)
                    {
                        // decrease jth parameter by EPSILON
                        double[] parametersMinus = new double[nParameters];
                        Array.Copy(parametersBackup, parametersMinus, nParameters);
                        parametersMinus[j] -= EPSILON;
                        network.Layers[iLayer].SetParameters(parametersMinus);
                        // then run network forward and compute loss
                        network.ForwardPass(iLayer, "end");
                        List<double[]> outputClassScoresMinus = network.OutputLayer.OutputClassScores;
                        double lossMinus = 0;
                        for (int m = 0; m < miniBatchSize; m++)
                        {
                            int trueLabel = trueLabels[m];
                            lossMinus -= Math.Log(outputClassScoresMinus[m][trueLabel]); // score of true class in example m
                        }
                        lossMinus /= miniBatchSize;

                        // increse jth parameter by EPSILON
                        double[] parametersPlus = new double[nParameters];
                        Array.Copy(parametersBackup, parametersPlus, nParameters);
                        parametersPlus[j] += EPSILON;
                        network.Layers[iLayer].SetParameters(parametersPlus);
                        // then run network forward and compute loss
                        network.ForwardPass(iLayer, "end");
                        List<double[]> outputClassScoresPlus = network.OutputLayer.OutputClassScores;
                        double lossPlus = 0;
                        for (int m = 0; m < miniBatchSize; m++)
                        {
                            int trueLabel = trueLabels[m];
                            lossPlus -= Math.Log(outputClassScoresPlus[m][trueLabel]); // score of true class in example m
                        }
                        lossPlus /= miniBatchSize;

                        // compute gradient numerically, trying to limit loss of significance!
                        //double orderOfMagnitude = Math.Floor(Math.Log10(lossPlus));
                        //lossPlus *= Math.Pow(10, -orderOfMagnitude);
                        //lossMinus *= Math.Pow(10, -orderOfMagnitude);
                        double gradientNumerical = (lossPlus - lossMinus) / (2*EPSILON);
                        //gradientNumerical *= Math.Pow(10, orderOfMagnitude);

                        // retrieve gradient computed with backprop
                        double gradientBackprop = parameterGradients[j];

                        //if (Math.Abs(gradientNumerical) > EPSILON || Math.Abs(gradientBackprop) > EPSILON) // when the gradient is very small, finite arithmetics effects are too large => don't check
                        //{
                            nChecks++;

                            // compare the gradients, again trying to limit loss of significance!
                            //orderOfMagnitude = Math.Floor(Math.Log10(Math.Abs(gradientNumerical)));
                            //double gradientNumericalRescaled = gradientNumerical * Math.Pow(10, -orderOfMagnitude);
                            //double gradientBackpropRescaled = gradientBackprop * Math.Pow(10, -orderOfMagnitude);
                            //double error = Math.Abs(gradientNumericalRescaled - gradientBackpropRescaled) * Math.Pow(10, orderOfMagnitude);
                            double error = Math.Abs(gradientNumerical - gradientBackprop);
                            double relativeError = error / Math.Max(Math.Abs(gradientNumerical), Math.Abs(gradientBackprop));
                            if (relativeError > MAX_RELATIVE_ERROR)
                            {
                                
                                Console.Write("\nGradient check failed for parameter {0}\n", j);
                                Console.WriteLine("\tBackpropagation gradient: {0}", gradientBackprop);
                                Console.WriteLine("\tFinite difference gradient: {0}", gradientNumerical);
                                Console.WriteLine("\tRelative error: {0}", relativeError);
                                
                                nErrors++;
                            }
                            cumulativeError = (relativeError + (nChecks - 1) * cumulativeError) / nChecks;
                        //}
                        
                        // restore original weights before checking next gradient
                        network.Layers[iLayer].SetParameters(parametersBackup);

                    }

                    if (nChecks == 0)
                        Console.Write("\nAll gradients are zero... Something is probably wrong!");
                    else if (nErrors == 0)
                    {
                        Console.Write("\nGradient check 100% passed!");
                        Console.Write("\nAverage error = {0}", cumulativeError);
                    }
                    else
                    {
                        Console.Write("\n{0} errors out of {1} checks.", nErrors, nChecks);
                        Console.Write("\nAverage error = {0}", cumulativeError);
                    }
                    Console.Write("\n\n");
                    Console.Write("Press any key to continue...");
                    Console.Write("\n\n");
                    Console.ReadKey();

                    // Now inputs
                    /*
                    nChecks = 0;
                    nErrors = 0;
                    cumulativeError = 0.0;

                    double[] inputBackup = network.Layers[iLayer].GetInput();
                    double[] inputGradients = network.Layers[iLayer].GetInputGradients();
                    int inputArraySize = inputBackup.Length;

                    Console.WriteLine("\n...with respect to INPUT");
                    for (int j = 0; j < inputArraySize; j++)
                    {
                        // decrease jth parameter by EPSILON
                        double[] inputMinus = new double[inputArraySize];
                        Array.Copy(inputBackup, inputMinus, inputArraySize);
                        inputMinus[j] -= EPSILON;
                        network.Layers[iLayer].SetInput(inputMinus);
                        // then run network forward and compute loss
                        network.ForwardPass(iLayer, "end");
                        List<double[]> outputClassScoresMinus = network.OutputLayer.OutputClassScores;
                        double lossMinus = 0;
                        for (int m = 0; m < miniBatchSize; m++)
                        {
                            int trueLabel = trueLabels[m];
                            lossMinus -= Math.Log(outputClassScoresMinus[m][trueLabel]); // score of true class in example m
                        }
                        lossMinus /= miniBatchSize;

                        // increse jth parameter by EPSILON
                        double[] inputPlus = new double[inputArraySize];
                        Array.Copy(inputBackup, inputPlus, inputArraySize);
                        inputPlus[j] += EPSILON;
                        network.Layers[iLayer].SetInput(inputPlus);
                        // then run network forward and compute loss
                        network.ForwardPass(iLayer, "end");
                        List<double[]> outputClassScoresPlus = network.OutputLayer.OutputClassScores;
                        double lossPlus = 0;
                        for (int m = 0; m < miniBatchSize; m++)
                        {
                            int trueLabel = trueLabels[m];
                            lossPlus -= Math.Log(outputClassScoresPlus[m][trueLabel]); // score of true class in example m
                        }
                        lossPlus /= miniBatchSize;

                        // compute gradient numerically
                        double gradientNumerical = (lossPlus - lossMinus) / (2*EPSILON);


                        // retrieve gradient computed with backprop
                        double gradientBackprop = inputGradients[j]/miniBatchSize;
                        // NOTE: it is divided by miniBatchSize because HERE the loss is defined as Loss / miniBatchSize

                        //if (Math.Abs(gradientNumerical) > EPSILON || Math.Abs(gradientBackprop) > EPSILON) // when the gradient is very small, finite arithmetics effects are too large => don't check
                        //{
                            nChecks++;

                            // compare the gradients
                            double relativeError = Math.Abs(gradientNumerical - gradientBackprop) / Math.Max(Math.Abs(gradientNumerical), Math.Abs(gradientBackprop));
                            if (relativeError > MAX_RELATIVE_ERROR)
                            {
                                
                                Console.Write("\nGradient check failed for input {0}\n", j);
                                Console.WriteLine("\tBackpropagation gradient: {0}", gradientBackprop);
                                Console.WriteLine("\tFinite difference gradient: {0}", gradientNumerical);
                                Console.WriteLine("\tRelative error: {0}", relativeError);
                                
                                nErrors++;
                            }
                            cumulativeError = (relativeError + (nChecks - 1) * cumulativeError) / nChecks;
                        //}

                        // restore original input before checking next gradient
                        network.Layers[iLayer].SetInput(inputBackup);

                    }

                    if (nChecks == 0)
                        Console.Write("\nAll gradients are zero... Something is probably wrong!");
                    else if (nErrors == 0)
                    {
                        Console.Write("\nGradient check 100% passed!");
                        Console.Write("\nAverage error = {0}", cumulativeError);
                    }
                    else
                    {
                        Console.Write("\n{0} errors out of {1} checks.", nErrors, nChecks);
                        Console.Write("\nAverage error = {0}", cumulativeError);
                    }
                    Console.Write("\n\n");
                    */
                }
            }
        }
    }

}
