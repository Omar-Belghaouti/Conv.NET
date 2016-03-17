using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    static class Utils
    {


        static Random random = new Random();

        /// <summary>
        /// Returns index of maximum element in the given input array.
        /// </summary>
        /// <param name="outputScores"></param>
        /// <returns></returns>
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


        /// <summary>
        /// Naive implementation of matrix-vector multiplication
        /// </summary>
        /// <param name="A"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static float[] MultiplyMatrixByVector(float[,] A, float[] b)
        {
            if (A.GetLength(1) != b.GetLength(0))
                throw new System.ArgumentException("Invalid matrix-vector multiplication.");

            float[] c = new float[A.GetLength(0)];

            for (int row = 0; row < A.GetLength(0); row++) {
                float sum = 0.0f;
                for (int col = 0; col < A.GetLength(1); col++)
                {
                    sum += A[row, col] * b[col];
                }
                c[row] = sum;
            }
            return c;
        }

        public static float[ , ] MultiplyMatrixByMatrix(float[ , ] A, float[ , ] B)
        {
            float[,] C = new float[A.GetLength(0), B.GetLength(1)];
            float sum = 0.0f;

            for (int row = 0; row < A.GetLength(0); row++)
            {
                sum = 0.0f;
                for (int col = 0; col < B.GetLength(1); col++)
                {
                    for (int k = 0; k < A.GetLength(1); k++)
                    {
                        sum += A[row, k] * B[k, col];
                    }
                    C[row, col] = sum;
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
        public static float[] MultiplyMatrixTranspByVector(float[,] A, float[] b)
        {
            if (A.GetLength(0) != b.GetLength(0))
                throw new System.ArgumentException("Invalid matrix^T-vector multiplication.");

            float[] c = new float[A.GetLength(1)];

            for (int row = 0; row < A.GetLength(1); row++)
            {
                float sum = 0.0f;
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
        public static int[] GenerateRandomPermutation(int maxN)
        {
            
            int[] sequence = new int[maxN];
            int n = maxN;

            for (int i = 0; i < maxN; i++)
                sequence[i] = i;
            
            while (n > 1)
            {
                int randomIndex = random.Next(n--);
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
                int randomIndex = random.Next(n--);
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
}
