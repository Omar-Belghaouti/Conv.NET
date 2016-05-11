using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using OpenCL.Net;

namespace JaNet
{

    public static class Global
    {
        public const double EPSILON = 0.000001;
        //public const int SEED = 2016;

        public static Random rng = new Random(); //(SEED);

        public static double RandomDouble()
        {
            return rng.NextDouble();
        }

        public static float RandomFloat()
        {
            return (float)rng.NextDouble();
        }
    }


    static class Utils
    {
        public static void SaveNetworkToFile(NeuralNetwork Network, string OutputFilePath)
        {
            
            // First prepare network for saving (copy all buffers to managed structures)
            for (int l = 1; l < Network.NumberOfLayers; l++)
            {
                Network.Layers[l].CopyBuffersToHost();
            }

            BinarySerialization.WriteToBinaryFile<NeuralNetwork>(OutputFilePath + Network.Name + ".bin", Network);
        
        }


        public static NeuralNetwork LoadNetworkFromFile(string InputFilePath, string networkName)
        {
            return BinarySerialization.ReadFromBinaryFile<NeuralNetwork>(InputFilePath + networkName + ".bin");
        }




#if TIMING_LAYERS

        public static Stopwatch InputFeedTimer = new Stopwatch();

        public static Stopwatch ConvForwardTimer = new Stopwatch();
        public static Stopwatch ConvBackpropTimer = new Stopwatch();
        public static Stopwatch ConvUpdateSpeedsTimer = new Stopwatch();
        public static Stopwatch ConvUpdateParametersTimer = new Stopwatch();
        public static Stopwatch ConvPadUnpadTimer = new Stopwatch();

        public static Stopwatch NonlinearityForwardTimer = new Stopwatch();
        public static Stopwatch NonlinearityBackpropTimer = new Stopwatch();

        public static Stopwatch PoolingForwardTimer = new Stopwatch();
        public static Stopwatch PoolingBackpropTimer = new Stopwatch();

        public static Stopwatch FCForwardTimer = new Stopwatch();
        public static Stopwatch FCBackpropTimer = new Stopwatch();
        public static Stopwatch FCUpdateSpeedsTimer = new Stopwatch();
        public static Stopwatch FCUpdateParametersTimer = new Stopwatch();

        public static Stopwatch BNFCForwardTimer = new Stopwatch();
        public static Stopwatch BNFCBackpropTimer = new Stopwatch();
        public static Stopwatch BNFCUpdateSpeedsTimer = new Stopwatch();
        public static Stopwatch BNFCUpdateParametersTimer = new Stopwatch();

        public static Stopwatch BNConvForwardTimer = new Stopwatch();
        public static Stopwatch BNConvBackpropTimer = new Stopwatch();
        public static Stopwatch BNConvUpdateSpeedsTimer = new Stopwatch();
        public static Stopwatch BNConvUpdateParametersTimer = new Stopwatch();

        public static Stopwatch SoftmaxTimer = new Stopwatch();

        public static void ResetTimers()
        {
            InputFeedTimer.Reset();

            ConvForwardTimer.Reset();
            ConvBackpropTimer.Reset();
            ConvUpdateSpeedsTimer.Reset();
            ConvUpdateParametersTimer.Reset();
            ConvPadUnpadTimer.Reset();

            NonlinearityForwardTimer.Reset();
            NonlinearityBackpropTimer.Reset();

            PoolingForwardTimer.Reset();
            PoolingBackpropTimer.Reset();

            FCForwardTimer.Reset();
            FCBackpropTimer.Reset();
            FCUpdateSpeedsTimer.Reset();
            FCUpdateParametersTimer.Reset();

            BNFCForwardTimer.Reset();
            BNFCBackpropTimer.Reset();
            BNFCUpdateSpeedsTimer.Reset();
            BNFCUpdateParametersTimer.Reset();

            BNConvForwardTimer.Reset();
            BNConvBackpropTimer.Reset();
            BNConvUpdateSpeedsTimer.Reset();
            BNConvUpdateParametersTimer.Reset();

            SoftmaxTimer.Reset();
        }
#endif

        /// <summary>
        /// Returns index of maximum element in the given input array.
        /// </summary>
        /// <param name="outputScores"></param>
        /// <returns></returns>
        public static int IndexOfMax(double[] outputScores)
        {
            int iMax = 0;
            double max = outputScores[0];
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

        /// <summary>
        /// Naive implementation of matrix-vector multiplication
        /// </summary>
        /// <param name="A"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double[] MultiplyMatrixByVector(double[,] A, double[] b)
        {
            if (A.GetLength(1) != b.GetLength(0))
                throw new System.ArgumentException("Invalid matrix-vector multiplication.");

            double[] c = new double[A.GetLength(0)];

            for (int row = 0; row < A.GetLength(0); row++) {
                double sum = 0.0f;
                for (int col = 0; col < A.GetLength(1); col++)
                {
                    sum += A[row, col] * b[col];
                }
                c[row] = sum;
            }
            return c;
        }

        public static double[,] MatrixMultiply(double[,] A, double[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != rowsB)
                throw new Exception("Non-conformable matrices in MatrixMultiply");

            double[,] C = new double[rowsA, colsB];

            for (int row = 0; row < rowsA; row++)
            {
                for (int col = 0; col < colsB; col++)
                {
                    for (int k = 0; k < colsA; k++)
                    {
                        C[row, col] += A[row, k] * B[k, col];
                    }
                }
            }
            return C;
        }


        /// <summary>
        /// Multiply transpose A^T of matrix A by vector b
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[] MultiplyMatrixTranspByVector(double[,] A, double[] b)
        {
            if (A.GetLength(0) != b.GetLength(0))
                throw new System.ArgumentException("Invalid matrix^T-vector multiplication.");

            double[] c = new double[A.GetLength(1)];

            for (int row = 0; row < A.GetLength(1); row++)
            {
                double sum = 0.0f;
                for (int col = 0; col < A.GetLength(0); col++)
                {
                    sum += A[col, row] * b[col];
                }
                c[row] = sum;
            }
            return c;
        }


        public static void SaveFilters(NeuralNetwork network, string outputFilePath)
        {
            if (network.Layers[1].Type != "Convolutional")
                throw new InvalidOperationException("First hidden layer is not convolutional. Cannot save filters.");

            Mem filtersGPU = network.Layers[1].WeightsGPU;

            int nFilters = network.Layers[1].OutputDepth;
            int inputDepth = network.Layers[1].InputDepth;
            int filterSize = network.Layers[1].FilterSize;

            int nParameters = nFilters * inputDepth * filterSize * filterSize;

            float[] filters = new float[nParameters];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                        filtersGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nParameters),
                                                        filters,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer filtersGPU");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            using (System.IO.StreamWriter outputFile = new System.IO.StreamWriter(outputFilePath))
            {
                foreach (float filterValue in filters)
                {
                    outputFile.WriteLine(filterValue.ToString());
                }
                Console.WriteLine("Filters of first convolutional layers saved in file" + outputFilePath);
            }

        }

        

    }

    public class Sequence
    {
        private int[] data;
        private int length;

        // Constructor
        public Sequence(int n)
        {
            data = new int[n];
            length = n;

            for (int i = 0; i < n; i++)
                data[i] = i;
        }

        // Indexer
        public int this[int i]
        {
            get {return data[i]; }
        }

        // Shuffle method
        public void Shuffle()
        {
            int i = length;

            while (i > 1)
            {
                int randomIndex = Global.rng.Next(i--);
                int tmp = data[randomIndex];
                data[randomIndex] = data[i];
                data[i] = tmp;
            }
        }

        // Get mini batch indices
        public int[] GetMiniBatchIndices(int iBeginning, int miniBatchSize)
        {
            int[] miniBatchIndices = new int[miniBatchSize];

            // If data.Length does not divide miniBatchSize, out of index exception can occur
            // In order to prevent this, the following check is performed.
            if (data.Length < iBeginning + miniBatchSize)
            {
                // If this occurs, read the remaining data... 
                for (int i = 0; i < data.Length - iBeginning; i++)
                    miniBatchIndices[i] = data[iBeginning + i];
                // ...and then resample some random data to complete the mini-batch
                for (int i = data.Length - iBeginning; i < miniBatchSize; i++)
                {
                    int iRandom = Global.rng.Next(data.Length);
                    miniBatchIndices[i] = data[iRandom];
                }
                // TODO: find a better solution! This is not an issue while training, but affects evaluation!
            }
            else
            {
                for (int i = 0; i < miniBatchSize; i++)
                    miniBatchIndices[i] = data[iBeginning + i];
            }
            return miniBatchIndices;
        }


    }
}
