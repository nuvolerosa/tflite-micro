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
// models data
#include "models/mars_e300_compressed_lstm.h"
#include "models/mars_e300_compressed_tcn.h"
#include "models/mars_e300_compressed_srnn.h"

int main() {
  printf("TFLM profiling runner\n");
  return 0;
}




// // ###################################################
// // namespaces
// // ###################################################

// namespace {
//   // Create alias for the class MicroMutableOpResolver
//   using OpResolver = tflite::MicroMutableOpResolver<8>;
//   // Registering minimal set of operations 
//   TfLiteStatus RegisterOps(OpResolver& op_resolver) {
//     TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
//     TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
//     TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
//     TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
//     TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
//     TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
//     TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
//     TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
//     return kTfLiteOk;
//   }
// } 

// // ###################################################
// // Define functions
// // ###################################################

// // Func (1): generate random input tensors (type:int8, size: 32x32x3)
// void GenerateRandomTensor(int8_t* tensor) {

//   // Create a random number generator for int8 values between -127 and 128
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_int_distribution<int> dist(-127, 128);
//   // Fill the tensor with random values
//   for (int i = 0; i < 32 * 32 * 3; ++i) {
//       tensor[i] = static_cast<int8_t>(dist(gen));  
//   }

// }

// // Func (2): profiling memory usage
// TfLiteStatus ProfileMemoryAndLatency(const uint8_t* g_model_data, int8_t* input_tensor) {

//   // Profiler and Ops resolver initialization
//   tflite::MicroProfiler profiler;
//   OpResolver op_resolver;
//   TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
//   // Defining Arena size by implementing 256kb constraint
//   constexpr int kTensorArenaSize = 256 * 1024;  
//   uint8_t tensor_arena[kTensorArenaSize];
//   // Defining space reserved to resource variables (WARNING: check what this is exactely and how to optimise)
//   constexpr int kNumResourceVariables = 1000; 
//   // Initialising Arena space allocator
//   tflite::RecordingMicroAllocator* allocator(
//     tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
//   // Initializing model interpreter 
//   tflite::RecordingMicroInterpreter interpreter(
//     tflite::GetModel(g_model_data), op_resolver, allocator,
//     tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
//     &profiler);
//   // Allocating memory
//   TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors()); 
//   // Checking that the model intakes 1 input tensor
//   TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);        
//   // Input is a pointer to the model input tensor (assumed to be type int8)
//   int8_t* input = interpreter.input(0)->data.int8;
//   // Check input is pointing "at something"
//   TFLITE_CHECK_NE(input, nullptr);
//   // Populate the input tensor with the generated random input data
//   for (int i = 0; i < 32 * 32 * 3; ++i) {
//       input[i] = input_tensor[i];  
//   }
//   // Running one inference step
//   TF_LITE_ENSURE_STATUS(interpreter.Invoke());
//   MicroPrintf("");  // Print an empty new line
//   profiler.LogTicksPerTagCsv();
//   MicroPrintf("");  // Print an empty new line
//   interpreter.GetMicroAllocator().PrintAllocations();

//   return kTfLiteOk;
// }

// // Func (3): get input, output, ops type and ops params sequentially during inference
// TfLiteStatus InferenceHistory4MACs(const uint8_t* g_model_data, int8_t* input_tensor) {

//   // Initialising op resolver
//   OpResolver op_resolver;
//   TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
//   // Defining Arena size by implementing 256kb constraint
//   constexpr int kTensorArenaSize = 256 * 1024;  
//   uint8_t tensor_arena[kTensorArenaSize];
//   constexpr int kNumResourceVariables = 1000; 
//   // Initialising Arena space allocator
//   tflite::RecordingMicroAllocator* allocator(
//     tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
//   // Initializing model interpreter 
//   tflite::RecordingMicroInterpreter interpreter(
//     tflite::GetModel(g_model_data), op_resolver, allocator,
//     tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables));
//   // Allocating memory
//   TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors()); 
//   // Checking that the model intakes 1 input tensor
//   TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);        
//   // Input is a pointer to the model input tensor (assumed to be type int8)
//   int8_t* input = interpreter.input(0)->data.int8;
//   // Check input is pointing "at something"
//   TFLITE_CHECK_NE(input, nullptr);
//   // Populate the input tensor with the generated random input data
//   for (int i = 0; i < 32 * 32 * 3; ++i) {
//       input[i] = input_tensor[i];  
//   }
//   // Running one inference step
//   TF_LITE_ENSURE_STATUS(interpreter.Invoke());
//   return kTfLiteOk;

// }

// int main(){
//   tflite::InitializeTarget();

//   // Generate random input
//   int8_t tensor[1 * 32 * 32 * 3];
//   GenerateRandomTensor(tensor);

//   // Latency Profiling (raw data only)
//   InferenceHistory4MACs(g_quant_model_0_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_1_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_2_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_3_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_4_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_5_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_6_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_7_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_8_model_data, tensor);
//   // InferenceHistory4MACs(g_quant_model_9_model_data, tensor);

//   // Memory Profiling
//   TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_0_model_data, tensor)); //no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_1_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_2_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_3_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_4_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_5_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_6_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_7_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_8_model_data, tensor)); // no core dump
//   // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(g_quant_model_9_model_data, tensor)); // core dumped
//   return 0;

// }


// // TBD: FUNC TO RUN INFERENCE AND CHECK ACCURACY
// // TfLiteStatus LoadQuantModelAndPerformInference(int8_t* input_tensor) {

// //   // Map the model into a usable data structure. This doesn't involve any
// //   // copying or parsing, it's a very lightweight operation.
// //   const tflite::Model* model =
// //       ::tflite::GetModel(g_quant_model_5_model_data);
// //   TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

// //   OpResolver op_resolver;
// //   TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

// //   // Arena size just a round number. The exact arena usage can be determined
// //   // using the RecordingMicroInterpreter.
// //   constexpr int kTensorArenaSize = 160 * 1024; // ~62% of Max MCU memory (256kb)
// //   uint8_t tensor_arena[kTensorArenaSize];

// //   tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
// //                                        kTensorArenaSize);
// //   TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

// //   // Retrive the memory space for one input tensor of the model
// //   TfLiteTensor* input = interpreter.input(0);
// //   TFLITE_CHECK_NE(input, nullptr);
// //   // Retrive the memory space for one output tensor of the model
// //   TfLiteTensor* output = interpreter.output(0);
// //   TFLITE_CHECK_NE(output, nullptr);

// //   // WARNING: they are all zero... why?
// //   // Extract information about the quantization parameters of the output tensor
// //   int8_t output_scale = output->params.scale;
// //   int8_t output_zero_point = output->params.zero_point;
// //   // Use the scale and zero-point for something
// //   MicroPrintf("Output scale: %f, Zero point: %d\n", output_scale, output_zero_point);

// //   // Make predictions
// //   // Populate the input tensor with the generated random input data
// //   int8_t* input_data = interpreter.input(0)->data.int8;
// //   for (int i = 0; i < 1 * 32 * 32 * 3; ++i) {
// //       input_data[i] = input_tensor[i];  
// //   }
// //   // Running one inference step
// //   TF_LITE_ENSURE_STATUS(interpreter.Invoke());
// //   int8_t y_pred = output->data.int8[0];
// //   MicroPrintf("Prediction: %d\n", y_pred);

// //   return kTfLiteOk;
// // }