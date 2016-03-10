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

    }
}
