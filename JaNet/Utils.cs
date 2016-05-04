using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

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



        /// <summary>
        /// Generates a random permutation of the set 0, 1, 2, ..., maxN-1 consecutive integers.
        /// </summary>
        /// <param name="maxN"></param>
        /// <returns></returns>
        [Obsolete("Replaced by Sequence class")]
        public static int[] GenerateRandomPermutation(int maxN)
        {
            
            int[] sequence = new int[maxN];
            int n = maxN;

            for (int i = 0; i < maxN; i++)
                sequence[i] = i;
            
            while (n > 1)
            {
                int randomIndex = Global.rng.Next(n--);
                int temp = sequence[randomIndex];
                sequence[randomIndex] = sequence[n];
                sequence[n] = temp;
            }

            return sequence;
        }

        /// <summary>
        /// Generate list of mini-batches indices, i.e. output is a list, each element of which is a minibatch, containing indices in random order
        /// </summary>
        /// <param name="maxN"></param>
        /// <returns></returns>
        [Obsolete("Method replaced by GenerateRandomPermutation")]
        public static List<int[]> GenerateMiniBatches(int maxN, int miniBatchSize)
        {

            if (maxN % miniBatchSize != 0)
                throw new System.ArgumentException("Cannot generate mini-batches.");


            List<int[]> miniBatchesList = new List<int[]>();
            int[] sequence = new int[maxN];
            int n = maxN;
            

            for (int i = 0; i < maxN; i++)
                sequence[i] = i;

            while (n > 1)
            {
                int randomIndex = Global.rng.Next(n--);
                int temp = sequence[randomIndex];
                sequence[randomIndex] = sequence[n];
                sequence[n] = temp;
            }

            int[] tmp = new int[miniBatchSize];
            int j = 0;
            for (int i = 0; i < maxN; i++)
            {
                tmp[j] = sequence[i];
                j++;

                if (j % miniBatchSize == 0)
                {
                    j = 0;
                    int[] tmp2 = (int[])tmp.Clone();
                    miniBatchesList.Add(tmp2);
                }
            }

            return miniBatchesList;
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
