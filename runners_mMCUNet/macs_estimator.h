// MACs estimation utilities for TFLite Micro models used in mMCUNet runners.
#ifndef MMCUNET_MACS_ESTIMATOR_H_
#define MMCUNET_MACS_ESTIMATOR_H_

#include <cstdint>

#include "tensorflow/lite/schema/schema_generated.h"

// Estimates the total number of MACs for a given TFLite flatbuffer model.
// This walks subgraph 0 and applies simple analytic formulas per supported op.
//
// Notes:
// - Some ops (e.g., Reshape, StridedSlice, Concatenation, Unpack, Pad, Quantize)
//   are treated as 0 MACs since they mainly move or transform data, not compute.
// - The estimate is approximate but sufficient for comparing architectures.
uint64_t EstimateModelMacs(const ::tflite::Model* model);

#endif  // MMCUNET_MACS_ESTIMATOR_H_

