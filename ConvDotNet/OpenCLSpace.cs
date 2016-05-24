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

    static class OpenCLSpace
    {

        #region Fields

        public static readonly int BASE_GROUP_SIZE = 32; // constant, depends on platform, e.g. use 32 for Nvidia (WARP) and 64 for AMD (WAVEFRONT)
        public static readonly int OPTIMAL_GROUP_SIZE = 128;// BASE_GROUP_SIZE * 4; // depends on device, e.g. 128 seems to be good for my GTX850M

        private static Context context;
        private static Device device;
        private static CommandQueue queue;

        private static List<int> maxWorkItemSizes;
        private static int maxWorkGroupSize;
        private static int maxComputeUnits;

        private static string kernelsPath;

        public static ErrorCode ClError;
        public static Event ClEvent;

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


        public static string KernelsPath
        {
            get { return kernelsPath; }
            set { kernelsPath = value; }
        }

        #endregion


        #region Kernels

        // Convolutional layer
        public static Kernel CreateRecFieldsLookupTable;
        public static Kernel CreatePaddingLookupTable;
        public static Kernel ZeroPad;
        public static Kernel ZeroUnpad;
        public static Kernel ConvForward;
        public static Kernel ConvBackPropagate;
        public static Kernel ConvUpdateSpeeds;
        public static Kernel ConvUpdateParameters;

        // Residual module
        public static Kernel SkipForward;
        public static Kernel SkipBackward;

        // MaxPooling layer
        public static Kernel CreateMaxPoolingTable;
        public static Kernel MaxPoolingForward;
        public static Kernel MaxPoolingBackward;

        // AveragePooling layer
        public static Kernel AveragePoolingForward;
        public static Kernel AveragePoolingBackward;

        // Fully connected layer
        public static Kernel FCUpdateParameters;
        public static Kernel FCForward;
        public static Kernel FCBackward;
        public static Kernel FCUpdateSpeeds;

        // ReLU layer
        public static Kernel ReLUForward;
        public static Kernel ReLUBackward;

        // ELU layer
        public static Kernel ELUForward;
        public static Kernel ELUBackward;

        // Tanh layer
        public static Kernel TanhForward;
        public static Kernel TanhBackward;

        // BatchNormFC layer
        public static Kernel BNFCComputeMeansVariances;
        public static Kernel BNFCForward;
        public static Kernel BNFCUpdateSpeeds;
        public static Kernel BNFCUpdateParameters;
        public static Kernel BNFCBackPropagate;

        // BatchNormConv layer
        public static Kernel BNConvComputeMeansVariances;
        public static Kernel BNConvForward;
        public static Kernel BNConvParameterGradientsBatch;
        public static Kernel BNConvUpdateSpeeds;
        public static Kernel BNConvUpdateParameters;
        public static Kernel BNConvBackPropagate;

        // Wipe kernels
        public static Kernel WipeBufferFloatKernel;
        public static Kernel WipeBufferIntKernel;
        public static Kernel WipeBufferBoolKernel;

        #endregion


        #region OpenCL setup

        public static void SetupSpace()
        {
            int deviceID; // will be asked to user

            List<Device> devicesList = new List<Device>();
            List<string> deviceNames = new List<string>();
            List<string> platformNames = new List<string>();
            int nDevices = 0;


            // Get list of available platforms
            Console.WriteLine("\nSearching for OpenCL-capable platforms... ");
            Platform[] platforms = Cl.GetPlatformIDs(out ClError);
            CheckErr(ClError, "CL.Setup: Cl.GetPlatformIDs");
            Console.WriteLine("{0} platforms found.\n", platforms.Length);

            foreach (Platform platform in platforms)
            {
                // Get platform info
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out ClError).ToString();
                Console.WriteLine("Platform: " + platformName);
                CheckErr(ClError, "CL.Setup: Cl.GetPlatformInfo");

                // Get all available devices within this platform and list them on screen
                foreach (Device dev in Cl.GetDeviceIDs(platform, DeviceType.All, out ClError))
                {
                    CheckErr(ClError, "CL.Setup: Cl.GetDeviceIDs");
                    string deviceName = Cl.GetDeviceInfo(dev, DeviceInfo.Name, out ClError).ToString();
                    CheckErr(ClError, "CL.Setup: Cl.GetDeviceInfo");
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
            context = Cl.CreateContext(null, 1, new[] { device }, ContextNotify, IntPtr.Zero, out ClError);    //Second parameter is amount of devices
            CheckErr(ClError, "CL.Setup: Cl.CreateContext");

            // Create OpenCL command queue
            queue = Cl.CreateCommandQueue(context, device, (CommandQueueProperties)0, out ClError);
            CheckErr(ClError, "CL.Setup: Cl.CreateCommandQueue");

            // Extract some device info
            maxWorkItemSizes = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, out ClError).CastToEnumerable<int>(new int[] { 0, 1, 2 }).ToList();
            Console.WriteLine("Max work item sizes: {0}, {1}, {2}", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            maxWorkGroupSize = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, out ClError).CastTo<int>();
            Console.WriteLine("Max work group size: {0}", maxWorkGroupSize);

            maxComputeUnits = Cl.GetDeviceInfo(device, DeviceInfo.MaxComputeUnits, out ClError).CastTo<int>();
            Console.WriteLine("Max compute units: {0}", maxComputeUnits);
        }


        #endregion


        #region Kernel loading and building

        public static Kernel LoadAndBuildKernel(string kernelFilePath, string kernelName)
        {
            // Attempt to read file
            if (!System.IO.File.Exists(kernelFilePath))
            {
                Console.WriteLine("Program doesn't exist at path " + kernelFilePath);
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            string kernelSource = System.IO.File.ReadAllText(kernelFilePath);

            // Create program
            OpenCL.Net.Program clProgram = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out ClError);
            CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.CreateProgramWithSource");

            //Compile kernel source
            ClError = Cl.BuildProgram(clProgram, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.BuildProgram " + kernelFilePath);

            //Check for any compilation errors
            if (Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Status, out ClError).CastTo<BuildStatus>()
                != BuildStatus.Success)
            {
                CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.GetProgramBuildInfo");
                Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                Console.WriteLine(Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Log, out ClError));
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            //Create the required kernel (entry function)
            Kernel kernel = Cl.CreateKernel(clProgram, kernelName, out ClError);
            CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.CreateKernel " + kernelName);

            return kernel;
        }


        public static void LoadKernels()
        {
            if (kernelsPath == null)
                throw new MissingFieldException("Path to kernels' source must be specified before calling LoadKernels()");

            // Convolutional layer
            CreateRecFieldsLookupTable = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "CreateRecFieldsLookupTable");
            CreatePaddingLookupTable = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "CreatePaddingLookupTable");
            ZeroPad = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "ZeroPad");
            ZeroUnpad = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "ZeroUnpad");
            ConvForward = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "ConvForward");
            ConvBackPropagate = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "ConvBackPropagate");
            ConvUpdateSpeeds = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "ConvUpdateSpeeds");
            ConvUpdateParameters = LoadAndBuildKernel(kernelsPath + "/Convolutional.cl", "ConvUpdateParameters");

            // Residual module
            SkipForward = LoadAndBuildKernel(kernelsPath + "/ResidualModule.cl", "SkipForward");
            SkipBackward = LoadAndBuildKernel(kernelsPath + "/ResidualModule.cl", "SkipBackward");

            // MaxPooling layer
            CreateMaxPoolingTable = LoadAndBuildKernel(kernelsPath + "/MaxPooling.cl", "CreateMaxPoolingTable");
            MaxPoolingForward = LoadAndBuildKernel(kernelsPath + "/MaxPooling.cl", "MaxPoolingForward");
            MaxPoolingBackward = LoadAndBuildKernel(kernelsPath + "/MaxPooling.cl", "MaxPoolingBackward");

            // AveragePooling layer
            AveragePoolingForward = LoadAndBuildKernel(kernelsPath + "/AveragePooling.cl", "AveragePoolingForward");
            AveragePoolingBackward = LoadAndBuildKernel(kernelsPath + "/AveragePooling.cl", "AveragePoolingBackward");

            // Fully connected layer
            FCUpdateParameters = LoadAndBuildKernel(kernelsPath + "/FullyConnected.cl", "FCUpdateParameters");
            FCForward = LoadAndBuildKernel(kernelsPath + "/FullyConnected.cl", "FCForward");
            FCBackward = LoadAndBuildKernel(kernelsPath + "/FullyConnected.cl", "FCBackward");
            FCUpdateSpeeds = LoadAndBuildKernel(kernelsPath + "/FullyConnected.cl", "FCUpdateSpeeds");

            // ReLU layer
            ReLUForward = LoadAndBuildKernel(kernelsPath + "/ReLU.cl", "ReLUForward");
            ReLUBackward = LoadAndBuildKernel(kernelsPath + "/ReLU.cl", "ReLUBackward");

            // ELU layer
            ELUForward = LoadAndBuildKernel(kernelsPath + "/ELU.cl", "ELUForward");
            ELUBackward = LoadAndBuildKernel(kernelsPath + "/ELU.cl", "ELUBackward");

            // Tanh layer
            TanhForward = LoadAndBuildKernel(kernelsPath + "/Tanh.cl", "TanhForward");
            TanhBackward = LoadAndBuildKernel(kernelsPath + "/Tanh.cl", "TanhBackward");

            // BatchNormFC
            BNFCComputeMeansVariances = LoadAndBuildKernel(kernelsPath + "/BatchNormFC.cl", "BNFCComputeMeansVariances");
            BNFCForward = LoadAndBuildKernel(kernelsPath + "/BatchNormFC.cl", "BNFCForward");
            BNFCUpdateSpeeds = LoadAndBuildKernel(kernelsPath + "/BatchNormFC.cl", "BNFCUpdateSpeeds");
            BNFCUpdateParameters = LoadAndBuildKernel(kernelsPath + "/BatchNormFC.cl", "BNFCUpdateParameters");
            BNFCBackPropagate = LoadAndBuildKernel(kernelsPath + "/BatchNormFC.cl", "BNFCBackPropagate");

            // BatchNormConv
            BNConvComputeMeansVariances = LoadAndBuildKernel(kernelsPath + "/BatchNormConv.cl", "BNConvComputeMeansVariances");
            BNConvForward = LoadAndBuildKernel(kernelsPath + "/BatchNormConv.cl", "BNConvForward");
            BNConvParameterGradientsBatch = LoadAndBuildKernel(kernelsPath + "/BatchNormConv.cl", "BNConvParameterGradientsBatch");
            BNConvUpdateSpeeds = LoadAndBuildKernel(kernelsPath + "/BatchNormConv.cl", "BNConvUpdateSpeeds");
            BNConvUpdateParameters = LoadAndBuildKernel(kernelsPath + "/BatchNormConv.cl", "BNConvUpdateParameters");
            BNConvBackPropagate = LoadAndBuildKernel(kernelsPath + "/BatchNormConv.cl", "BNConvBackPropagate");

            // Wipe kernel
            WipeBufferFloatKernel = LoadAndBuildKernel(kernelsPath + "/Wipe.cl", "WipeBufferFloatKernel");
            WipeBufferIntKernel = LoadAndBuildKernel(kernelsPath + "/Wipe.cl", "WipeBufferIntKernel");
            WipeBufferBoolKernel = LoadAndBuildKernel(kernelsPath + "/Wipe.cl", "WipeBufferBoolKernel");
        }


        #endregion


        #region Helper methods

        public static void WipeBuffer(Mem buffer, int nElementsInBuffer, Type type)
        {
            Kernel WipeKernel;

            if (type == typeof(float))
                WipeKernel = WipeBufferFloatKernel;
            else if (type == typeof(int))
                WipeKernel = WipeBufferIntKernel;
            else if (type == typeof(bool))
                WipeKernel = WipeBufferBoolKernel;
            else
                throw new ArgumentException("Type not supported. Use either float, int, or bool.");

            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(WipeKernel, 0, buffer);
            OpenCLSpace.ClError |= Cl.SetKernelArg(WipeKernel, 1, (IntPtr)sizeof(int), nElementsInBuffer);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg WipeBufferKernel");

            // Work sizes
            IntPtr[] localWorkSizePtr = { (IntPtr)OPTIMAL_GROUP_SIZE };
            IntPtr[] globalWorkSizePtr = { (IntPtr)(OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nElementsInBuffer) / (double)OPTIMAL_GROUP_SIZE)) };

            // Run kernel
            ClError = Cl.EnqueueNDRangeKernel(queue,
                                                WipeKernel,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out ClEvent);
            CheckErr(ClError, "Cl.EnqueueNDRangeKernel ZeroUnpadBatch");

            ClError = Cl.ReleaseEvent(ClEvent);
            CheckErr(ClError, "Cl.ReleaseEvent");

            ClError = Cl.Finish(queue);
            CheckErr(ClError, "Cl.Finish");

            //Cl.ReleaseKernel(WipeKernel);
        }

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

        #endregion

    }

}