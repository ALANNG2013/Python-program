# Title: Python program simple test
# Application: Eclispe cpp 2019-12, Windows10, Anaconda3
# Date: 9/02/2020
# Author : ALAN NG, OUHK DL Student
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#from tensorflow.python.client import device_lib
import keras
import sys, os
import time
import threading
import multiprocessing
import datetime
import pytest
import scipy as sp
import ctypes
import numba
from numba import cuda
import pycuda.autoinit
import pathlib
from pathlib import Path
import imp
import numpy
import matplotlib.pyplot as plt
import platform,socket

#from tensorflow.python.platform.test import is_built_with_cuda
from pycuda.compiler import _find_pycuda_include_path
from numba.cuda.tests.cudasim.support import cuda_module_in_device_function

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("--------  Tensorflow information ----------")
#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)
#print("Tensorflow Module", tf._current_module)
#print("Tensorflow site-packages DIR: ", tf._site_packages_dirs, "\n")
print("Tensorflow version = ", tf.version)
print(tf.is_tensor)
print("Tensorflow File: ", tf.__file__)
print("Tensorflow path: ", tf.__path__)
print('\n'.join(sorted(sys.path)))
print(('Is your GPU available for use?\n{0}').format(
    'Yes, your GPU is available: True' if tf.test.is_gpu_available() == True else 'No, your GPU is NOT available: False'
))
print("---------------------------------------------")
print("--------------- Python information -------------")
major = sys.version_info.major
minor = sys.version_info.minor
print("    Python version = py{}.{}".format(major, minor))
print("    Python version: ", sys.version)
# print(tf.python_io)
print("-------------------------------------------------")
print("----------------  other application -----------------")
print("Keras version = ", keras.__version__ )
print("Scipy version = ", sp.version.version)
print("numba version = ", numba.__version__)
print("pytest install -> ")
print(pytest)
print("pycuda install -> ", pycuda)
print(pycuda.autoinit)
print(_find_pycuda_include_path)
print("----------------------------------------------------------") 
print("--------------  CUDA information ----------------")
print("If you do have a CUDA-enabled GPU on your system, you should see a message like <Managed Device 0> : ", cuda.gpus, "\n")
env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if env:
    print(len(env.split(',')), " <-- device found non-empty CUDA_VISIBLE_DEVICES.")
print("CUDA device: ", cuda_module_in_device_function)
#print("CUDA install: ", is_built_with_cuda)
#print("CUDA connect: ", tf.test.is_built_with_cuda())
print("----------------------------------------------------")
print("--------  GPU information ------------------")
device_to_use = "gpu"
if device_to_use == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
with tf.device(device_name):
    #print(device_lib.list_local_devices())
    #print("GPU connection: ", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))  
    #print(tf.test.is_gpu_available)    
    #print(tf.test.gpu_device_name())
    print("Device to use = ", device_name)
print("----------------------------------------------")

print("------- System information --------")
current_day = (datetime.datetime.now().strftime("%Y_%m_%d_%H_%S"))
result_dir = os.path.join(os.path.dirname(__file__), " Today : date & time ", current_day)
print("Current workspace = ", result_dir)
#print("Include file: ", tf.sysconfig.get_include())
#print("Library file: ", tf.sysconfig.get_lib(), "\n")
print("Current path: ", os.path.dirname(os.path.abspath(__file__)))
#print("current path:", os.getcwd())
print("Upper directory from current one: ", os.chdir(os.path.dirname(os.getcwd())))
#Returns the path of the directory, where your script file is placed
mypath = Path().absolute()
print('Absolute path : {}'.format(mypath))
#if you want to go to any other file inside the subdirectories of the directory path got from above method
filePath = mypath/'testprog'/'helloworld.py'
print('File path : {}'.format(filePath))
#To check if file present in that directory or Not
isfileExist = filePath.exists()
print('isfileExist : {}'.format(isfileExist))
#To check if the path is a directory or a File
isadirectory = filePath.is_dir()
print('isadirectory : {}'.format(isadirectory))
#To get the extension of the file
fileExtension = mypath/'testprog'/'helloworld.py'
print('File extension : {}'.format(filePath.suffix))

current_dir = pathlib.Path(__file__).parent
current_file = pathlib.Path(__file__)
print("current DIR: ", current_dir)
print("current file: ", current_file, "\n")

try:
# import tensorflow as tf
    print("TensorFlow successfully installed.")
#    if tf.test.is_built_with_cuda():
#        print("The installed version of TensorFlow includes GPU support.")
#    else:
#        print("The installed version of TensorFlow does not include GPU support.")
    #sys.exit(0)
except ImportError:
    print("ERROR: Failed to import the TensorFlow module.")

    candidate_explanation = False

    python_version = sys.version_info.major, sys.version_info.minor
    print("\n- Python version is %d.%d." % python_version)
    if not (python_version == (3, 6) or python_version == (3, 7)):
        candidate_explanation = True
        print("- The official distribution of TensorFlow for Windows requires "
          "Python version 3.6 or 3.7.")
try:
    _, pathname, _ = imp.find_module("tensorflow")
    print("\n- TensorFlow is installed at: %s" % pathname)
except ImportError:
    candidate_explanation = False
    print("""
- No module named TensorFlow is installed in this Python environment. You may
  install it using the command `pip install tensorflow`.""")

try:
    msvcp140 = ctypes.WinDLL("msvcp140.dll")
except OSError:
    candidate_explanation = True
    print("""
- Could not load 'msvcp140.dll'. TensorFlow requires that this DLL be
  installed in a directory that is named in your %PATH% environment
  variable. You may install this DLL by downloading Microsoft Visual
  C++ 2015 Redistributable Update 3""")

try:
    cudart64_80 = ctypes.WinDLL("cudart64_80.dll")
except OSError:
    candidate_explanation = True
    print("""
- Could not load 'cudart64_80.dll'. The GPU version of TensorFlow
  requires that this DLL be installed in a directory that is named in
  your %PATH% environment variable. Download and install CUDA 8.0""")

try:
    nvcuda = ctypes.WinDLL("nvcuda.dll")
except OSError:
    candidate_explanation = True
    print("""
- Could not load 'nvcuda.dll'. The GPU version of TensorFlow requires that
  this DLL be installed in a directory that is named in your %PATH%
  environment variable. Typically it is installed in 'C:\Windows\System32'.
  If it is not present, ensure that you have a CUDA-capable GPU with the
  correct driver installed.""")

cudnn5_found = False
try:
    cudnn5 = ctypes.WinDLL("cudnn64_5.dll")
    cudnn5_found = True
except OSError:
    candidate_explanation = True
    print(" \n ----> cudnn5 not found <----")

cudnn6_found = False
try:
    cudnn = ctypes.WinDLL("cudnn64_6.dll")
    cudnn6_found = True
    if cudnn6_found:
        print(" \n ---> cudnn6 found  <---")
except OSError:
    candidate_explanation = True
    if not cudnn5_found or not cudnn6_found:
        print("Both cudnn5 and cudnn6 not found")
    if not cudnn5_found and not cudnn6_found:
        print("- Could not find cuDNN.")
    elif not cudnn5_found:
        print("- Could not find cuDNN 5.1.")
    else:
        print("- Could not find cuDNN 6.")
        print("""
  The GPU version of TensorFlow requires that the correct cuDNN DLL be installed
  in a directory that is named in your %PATH% environment variable. Note that
  installing cuDNN is a separate step from installing CUDA, and it is often
  found in a different directory from the CUDA DLLs. The correct version of
  cuDNN depends on your version of TensorFlow.""")

cudnn7_found = False
try:
    cudnn = ctypes.WinDLL("cudnn64_7.dll")
    cudnn7_found = True
    if cudnn7_found:
        print(" ---> cudnn7 found <---")
except OSError:
    candidate_explanation = True

    if not cudnn6_found or not cudnn7_found:
        print()
    if not cudnn6_found and not cudnn7_found:
        print("- Could not find cuDNN.")
    elif not cudnn6_found:
        print("- Could not find cuDNN 6.1.")
    else:
        print("- Could not find cuDNN 7.")
        print("""
  The GPU version of TensorFlow requires that the correct cuDNN DLL be installed
  in a directory that is named in your %PATH% environment variable. Note that
  installing cuDNN is a separate step from installing CUDA, and it is often
  found in a different directory from the CUDA DLLs. The correct version of
  cuDNN depends on your version of TensorFlow:

  * TensorFlow 1.2.1 or earlier requires cuDNN 6.1. ('cudnn64_6.dll')
  * TensorFlow 1.3 or later requires cuDNN 6. ('cudnn64_7.dll')

  You may install the necessary DLL by downloading cuDNN""")

    if not candidate_explanation:
        print("""
- All required DLLs appear to be present. Please open an issue on the
  TensorFlow GitHub page""")
        
NUM_WORK = os.cpu_count() 
print("")
print("----- Systenm platform information -----")
print("Platform node = ", platform.node())
print("Python version = ", sys.version)
print("SYSTEM platform = ", sys.platform)
print("SYSTEM platform = ", platform.machine())
print("Architecture: " + platform.architecture()[0])
print("CPU number = ", multiprocessing.cpu_count())
print("Platform release = ", platform.release())
print("Platform version =", platform.version())
print("Platform system = ", platform.system())
print(platform.platform())
print("Platform processor = ", platform.processor())
print(platform.uname())
print("Computer name = ", socket.gethostname())
print("Computer IP address = ", socket.gethostbyname(socket.gethostname()))
print("Processors: ")

def crunch_numbers():
    """ Do some computations """
    print("PID: %s, Process Name: %s, Thread Name: %s" % (
        os.getpid(),
        multiprocessing.current_process().name,
        threading.current_thread().name)
    )
    x = 0
    ii = 1
    while ii < 1000000:
        x*x
        ii+=1
       
## Run tasks serially 
start_time = time.time()
for _ in range(NUM_WORK):
    crunch_numbers()
    end_time = time.time()
print("Serial time=", end_time - start_time)

# Run tasks using threads
start_time = time.time()
threads = [threading.Thread(target=crunch_numbers) for _ in range(NUM_WORK)]
[thread.start() for thread in threads]
[thread.join() for thread in threads]
end_time = time.time()
print("Threads time=", end_time - start_time)

print("Draw a polynomial regression line through the data points ")
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

A_model = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

A_line = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(A_line, A_model(A_line))
plt.show()
