/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tflite-micro Apache License, Version 2.0, modified by @n-rosi

// ###################################################
// Import dependencies
// ###################################################

// libs
#include <iostream>
#include <random>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>    
// tflite libs
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
// tflite-micro libs
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
// macs estimator
#include "runners_mMCUNet/macs_estimator.h"
// models data
#include "models/mars_e300_comp_bgaf_watt_lstm.h"
#include "models/mars_e300_comp_bgf_watt_lstm.h"
#include "models/mars_e300_comp_gf_watt_lstm.h"
#include "models/mars_e300_comp_gf_woatt_lstm.h"
#include "models/mars_e300_comp_bgaf_watt_tcn.h"
#include "models/mars_e300_comp_bgf_watt_tcn.h"
#include "models/mars_e300_comp_gf_watt_tcn.h"
#include "models/mars_e300_comp_gf_woatt_tcn.h"
#include "models/mars_e300_comp_bgaf_watt_srnn.h"
#include "models/mars_e300_comp_bgf_watt_srnn.h"
#include "models/mars_e300_comp_gf_watt_srnn.h"
#include "models/mars_e300_comp_gf_woatt_srnn.h"
#include "models/mymodel_int8.h"

// ###################################################
// namespaces
// ###################################################

namespace {
  // Create alias for the class MicroMutableOpResolver
  using OpResolver = tflite::MicroMutableOpResolver<19>;
  // Registering minimal set of operations 
  TfLiteStatus RegisterOps(OpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
    TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice()); 
    TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation()); 
    TF_LITE_ENSURE_STATUS(op_resolver.AddSum());
    TF_LITE_ENSURE_STATUS(op_resolver.AddUnpack());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSplit());
    TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
    TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
    TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
    TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
    TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
    return kTfLiteOk;
  }
} 

// ###################################################
// Define functions
// ###################################################

// Func (1): generate random input tensors (type:int8, size: 32x32x3)
void GenerateRandomTensor(int8_t* tensor) {

  // Create a random number generator for int8 values between -127 and 128
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-127, 128);
  // Fill the tensor with random values
  for (int i = 0; i < 32 * 64 * 3; ++i) {
      tensor[i] = static_cast<int8_t>(dist(gen));  
  }

}

// Func (2): profiling memory usage
TfLiteStatus ProfileMemoryAndLatency(const uint8_t* g_model_data, int8_t* input_tensor) {

  // Profiler and Ops resolver initialization
  tflite::MicroProfiler profiler;
  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  // Defining Arena size by implementing 512kb constraint
  constexpr int kTensorArenaSize = 512 * 1024;  
  uint8_t tensor_arena[kTensorArenaSize];
  // Defining space reserved to resource variables (WARNING: check what this is exactely and how to optimise)
  constexpr int kNumResourceVariables = 1000; 
  // Initialising Arena space allocator
  tflite::RecordingMicroAllocator* allocator(
    tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  // Initializing model interpreter 
  tflite::RecordingMicroInterpreter interpreter(
    tflite::GetModel(g_model_data), op_resolver, allocator,
    tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
    &profiler);
  // Allocating memory
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors()); 
  // Checking that the model intakes 1 input tensor
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);        
  // Input is a pointer to the model input tensor (assumed to be type int8)
  int8_t* input = interpreter.input(0)->data.int8;
  // Check input is pointing "at something"
  TFLITE_CHECK_NE(input, nullptr);
  // Populate the input tensor with the generated random input data
  for (int i = 0; i < 32 * 64 * 3; ++i) {
      input[i] = input_tensor[i];  
  }
  // Estimate MACs using the flatbuffer model.
  const tflite::Model* model = tflite::GetModel(g_model_data);
  const uint64_t macs = EstimateModelMacs(model);
  MicroPrintf("Estimated MACs per inference: %llu",
              static_cast<unsigned long long>(macs));
  // Running one inference step
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();
  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();

  return kTfLiteOk;
}


int main(){
  tflite::InitializeTarget();

  // Generate random input
  int8_t tensor[1 * 32 * 64 * 6];
  GenerateRandomTensor(tensor);

  std::string log_folder = "/home/nrosi/GitHub/mMCUNet/externals/tflite-micro/runners_mMCUNet/logs/";
  std::string log_path;

  // Lateny and Memory Profiling
  // log_path = log_folder + "mars_e300_comp_bgf_watt_lstm.txt";
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency( mars_e300_comp_bgf_watt_lstm, tensor));
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mars_e300_comp_bgf_watt_tcn, tensor));
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency( mars_e300_comp_bgf_watt_srnn, tensor));

  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mars_e300_comp_gf_watt_lstm, tensor));
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mars_e300_comp_gf_watt_tcn, tensor));
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mars_e300_comp_gf_watt_srnn, tensor));

  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mars_e300_comp_gf_woatt_lstm, tensor));
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mars_e300_comp_gf_woatt_tcn, tensor));
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mars_e300_comp_gf_woatt_srnn, tensor));

  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(mymodel_int8, tensor));
  return 0;

}

