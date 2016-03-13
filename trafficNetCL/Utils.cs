using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    static class Utils
    {
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
                    sum += A[row, col] * b[row];
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
                    sum += A[col, row] * b[row];
                }
                c[row] = sum;
            }
            return c;
        }

    }
}
