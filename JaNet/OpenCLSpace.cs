using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using OpenCL.Net;
using OpenCL.Net.Extensions;


namespace JaNet
{

#if OPENCL_ENABLED

    static class OpenCLSpace
    {
        public static Event NewEvent;
        public static ErrorCode NewClError;

        #region Fields

        private static Context context;
        private static Device device;
        private static CommandQueue queue;

        private static List<int> maxWorkItemSizes;
        private static int maxWorkGroupSize;
        private static int maxComputeUnits;

        private static string kernelsPath;

        #endregion


        #region Properties

        public static Context Context
        {
            get { return context; }
            set { context = value; }
        }

        public static Device Device
        {
            get { return device; }
        }

        public static CommandQueue Queue
        {
            get { return queue; }
            set { queue = value; }
        }

        public static List<int> MaxWorkItemSizes
        {
            get { return maxWorkItemSizes; }
        }

        public static int MaxWorkGroupSize
        {
            get { return maxWorkGroupSize; }
        }

        public static int MaxComputeUnits
        {
            get { return maxComputeUnits; }
        }

        /*
        public static ErrorCode ClError
        {
            get { return clError; }
            set { clError = value; }
        }
        public static Event ClEvent
        {
            get { return clEvent; }
            set { clEvent = value; }
        }
         * */

        public static string KernelsPath
        {
            get { return kernelsPath; }
            set { kernelsPath = value; }
        }

        #endregion


        #region Kernels

        // FullyConnected layer
        public static Kernel FCForward;
        public static Kernel FCBackward;
        public static Kernel FCUpdateParameters;

        // ReLU layer
        public static Kernel ReLUForward;
        public static Kernel ReLUBackward;

        // Softmax layer
        public static Kernel SoftmaxForward;

        // Cross-entropy gradient
        public static Kernel CrossEntropyGradient;

        // Classification
        public static Kernel CheckClassification;

        #endregion


        #region OpenCL setup

        public static void SetupSpace()
        {
            int deviceID; // will be asked to user

            List<Device> devicesList = new List<Device>();
            List<string> deviceNames = new List<string>();
            List<string> platformNames = new List<string>();
            int nDevices = 0;

            ErrorCode clError;

            // Get list of available platforms
            Console.WriteLine("\nSearching for OpenCL-capable platforms... ");
            Platform[] platforms = Cl.GetPlatformIDs(out clError);
            CheckErr(clError, "CL.Setup: Cl.GetPlatformIDs");
            Console.WriteLine("{0} platforms found.\n", platforms.Length);

            foreach (Platform platform in platforms)
            {
                // Get platform info
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out clError).ToString();
                Console.WriteLine("Platform: " + platformName);
                CheckErr(clError, "CL.Setup: Cl.GetPlatformInfo");

                // Get all available devices within this platform and list them on screen
                foreach (Device dev in Cl.GetDeviceIDs(platform, DeviceType.All, out clError))
                {
                    CheckErr(clError, "CL.Setup: Cl.GetDeviceIDs");
                    string deviceName = Cl.GetDeviceInfo(dev, DeviceInfo.Name, out clError).ToString();
                    CheckErr(clError, "CL.Setup: Cl.GetDeviceInfo");
                    Console.WriteLine("Device {0}: {1}", nDevices, deviceName);

                    devicesList.Add(dev);
                    deviceNames.Add(deviceName);
                    platformNames.Add(platformName);
                    nDevices++;
                }
                Console.WriteLine();
            }

            if (nDevices == 0)
            {
                throw new PlatformNotSupportedException("No OpenCL-capable platform and/or devices were found on this system.");
            }

            Console.Write("Enter ID of device to use: ");
            deviceID = int.Parse(Console.ReadLine()); 

            // Select device according to user's choice
            device = devicesList[deviceID];
            Console.WriteLine("\nUsing device {0}", deviceNames[deviceID]);

            // Create OpenCL context
            context = Cl.CreateContext(null, 1, new[] { device }, ContextNotify, IntPtr.Zero, out clError);    //Second parameter is amount of devices
            CheckErr(clError, "CL.Setup: Cl.CreateContext");

            // Create OpenCL command queue
            queue = Cl.CreateCommandQueue(context, device, (CommandQueueProperties)0, out clError);
            CheckErr(clError, "CL.Setup: Cl.CreateCommandQueue");

            // Extract some device info
            maxWorkItemSizes = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, out clError).CastToEnumerable<int>(new int[] { 0, 1, 2 }).ToList();
            Console.WriteLine("Max work item sizes: {0}, {1}, {2}", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            maxWorkGroupSize = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, out clError).CastTo<int>();
            Console.WriteLine("Max work group size: {0}", maxWorkGroupSize);

            maxComputeUnits = Cl.GetDeviceInfo(device, DeviceInfo.MaxComputeUnits, out clError).CastTo<int>();
            Console.WriteLine("Max compute units: {0}", maxComputeUnits);

        }


        #endregion


        #region Kernel loading and building

        public static Kernel LoadAndBuildKernel(string kernelFilePath, string kernelName)
        {
            ErrorCode clError;

            // Attempt to read file
            if (!System.IO.File.Exists(kernelFilePath))
            {
                Console.WriteLine("Program doesn't exist at path " + kernelFilePath);
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            string kernelSource = System.IO.File.ReadAllText(kernelFilePath);

            // Create program
            OpenCL.Net.Program clProgram = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out clError);
            CheckErr(clError, "CL.LoadAndBuildKernel: Cl.CreateProgramWithSource");

            //Compile kernel source
            clError = Cl.BuildProgram(clProgram, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            CheckErr(clError, "CL.LoadAndBuildKernel: Cl.BuildProgram");

            //Check for any compilation errors
            if (Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Status, out clError).CastTo<BuildStatus>()
                != BuildStatus.Success)
            {
                CheckErr(clError, "CL.LoadAndBuildKernel: Cl.GetProgramBuildInfo");
                Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                Console.WriteLine(Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Log, out clError));
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            //Create the required kernel (entry function)
            Kernel kernel = Cl.CreateKernel(clProgram, kernelName, out clError);
            CheckErr(clError, "CL.LoadAndBuildKernel: Cl.CreateKernel");

            return kernel;
        }


        public static void LoadKernels()
        {
            if (kernelsPath == null)
                throw new MissingFieldException("Path to kernels' source must be specified before calling LoadKernels()");

            // Fully connected layer
            FCForward = LoadAndBuildKernel(kernelsPath + "/FCForward.cl", "FCForward");
            FCBackward = LoadAndBuildKernel(kernelsPath + "/FCBackward.cl", "FCBackward");
            FCUpdateParameters = LoadAndBuildKernel(kernelsPath + "/FCUpdateParameters.cl", "FCUpdateParameters");

            // ReLU layer
            ReLUForward = LoadAndBuildKernel(kernelsPath + "/ReLUForward.cl", "ReLUForward");
            ReLUBackward = LoadAndBuildKernel(kernelsPath + "/ReLUBackward.cl", "ReLUBackward");

            // Softmax layer
            SoftmaxForward = LoadAndBuildKernel(kernelsPath + "/SoftmaxForward.cl", "SoftmaxForward");
            
            // Cross-entropy gradient
            CrossEntropyGradient = LoadAndBuildKernel(kernelsPath + "/CrossEntropyGradient.cl", "CrossEntropyGradient");


            // Classification
            // TODO: implement a better kernel
            CheckClassification = LoadAndBuildKernel(kernelsPath + "/CheckClassification.cl", "CheckClassification");
        }

        #endregion


        #region Helper methods

        public static void CheckErr(ErrorCode err, string name)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("ERROR: " + name + " (" + err.ToString() + ")");
                Console.ReadKey();
            }
        }

        public static void ContextNotify(string errInfo, byte[] data, IntPtr cb,
            IntPtr userData)
        {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        // TODO: delete this if not needed
        /*
        public static void ClearBuffer(Mem buffer, int bufferSize)
        {
            float[] zeros = new float[bufferSize];
            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(   OpenCLSpace.Queue,
                                                buffer,
                                                OpenCL.Net.Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(sizeof(float) * bufferSize),
                                                zeros,
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "CL.ClearBuffer: Cl.EnqueueWriteBuffer");
        }
        */


        #endregion

    }

#endif
}
