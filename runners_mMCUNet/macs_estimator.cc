// Implementation of MACs estimation for TFLite Micro models.

#include "runners_mMCUNet/macs_estimator.h"

#include <cstddef>
#include <cstdio>

namespace {

// Safely returns the product of all dims in a tensor shape.
int64_t NumElements(const ::tflite::Tensor* tensor) {
  if (tensor == nullptr || tensor->shape() == nullptr) {
    return 0;
  }
  int64_t prod = 1;
  for (int i = 0; i < tensor->shape()->Length(); ++i) {
    prod *= tensor->shape()->Get(i);
  }
  return prod;
}

inline int64_t GetDim(const ::tflite::Tensor* tensor, int idx, int64_t default_value = 1) {
  if (tensor == nullptr || tensor->shape() == nullptr) {
    return default_value;
  }
  if (idx < 0 || idx >= tensor->shape()->Length()) {
    return default_value;
  }
  return tensor->shape()->Get(idx);
}

// Estimates MACs for a Conv2D op using output and filter shapes.
int64_t EstimateConv2DMacs(const ::tflite::Tensor* input,
                           const ::tflite::Tensor* filter,
                           const ::tflite::Tensor* output) {
  (void)input;
  if (filter == nullptr || output == nullptr || filter->shape() == nullptr ||
      output->shape() == nullptr) {
    return 0;
  }

  // TFLite Conv2D filter shape is typically [Kh, Kw, Cin, Cout].
  const int64_t kh = GetDim(filter, 0, 1);
  const int64_t kw = GetDim(filter, 1, 1);
  const int64_t cin = GetDim(filter, 2, 1);
  const int64_t cout = GetDim(filter, 3, 1);

  // Output shape [N, Hout, Wout, Cout] (NHWC).
  const int64_t n = GetDim(output, 0, 1);
  const int64_t hout = GetDim(output, 1, 1);
  const int64_t wout = GetDim(output, 2, 1);

  return n * hout * wout * cout * kh * kw * cin;
}

// Estimates MACs for a FullyConnected op.
int64_t EstimateFullyConnectedMacs(const ::tflite::Tensor* input,
                                   const ::tflite::Tensor* weights,
                                   const ::tflite::Tensor* output) {
  if (input == nullptr || weights == nullptr || output == nullptr) {
    return 0;
  }

  const int64_t input_elems = NumElements(input);
  const int64_t output_elems = NumElements(output);

  // Debug prints for FC layer dimensions.
  std::printf("FullyConnected input_elems: %lld\n",
              static_cast<long long>(input_elems));
  std::printf("FullyConnected output_elems: %lld\n",
              static_cast<long long>(output_elems));

  // Each output element is roughly a dot-product over the input.
  return input_elems * output_elems;
}

// Estimates MACs for reduction ops like MEAN or SUM.
int64_t EstimateReduceMacs(const ::tflite::Tensor* input,
                           const ::tflite::Tensor* output) {
  if (input == nullptr || output == nullptr) {
    return 0;
  }
  // A simple approximation: one accumulation per input element.
  return NumElements(input);
}

}  // namespace

uint64_t EstimateModelMacs(const ::tflite::Model* model) {
  if (model == nullptr || model->subgraphs() == nullptr ||
      model->subgraphs()->size() == 0) {
    return 0;
  }

  const auto* subgraph = model->subgraphs()->Get(0);
  if (subgraph == nullptr || subgraph->operators() == nullptr ||
      subgraph->tensors() == nullptr) {
    return 0;
  }

  const auto* opcodes = model->operator_codes();
  if (opcodes == nullptr) {
    return 0;
  }

  uint64_t total_macs = 0;

  for (uint32_t i = 0; i < subgraph->operators()->size(); ++i) {
    const auto* op = subgraph->operators()->Get(i);
    if (op == nullptr) {
      continue;
    }

    const uint32_t opcode_index = op->opcode_index();
    if (opcode_index >= opcodes->size()) {
      continue;
    }

    const auto* opcode = opcodes->Get(opcode_index);
    if (opcode == nullptr) {
      continue;
    }

    const auto builtin_code =
        static_cast<::tflite::BuiltinOperator>(opcode->builtin_code());

    const auto* tensors = subgraph->tensors();
    const ::tflite::Tensor* input0 = nullptr;
    const ::tflite::Tensor* input1 = nullptr;
    const ::tflite::Tensor* output0 = nullptr;

    if (op->inputs() != nullptr && op->inputs()->size() > 0) {
      const int32_t idx0 = op->inputs()->Get(0);
      if (idx0 >= 0 && idx0 < tensors->size()) {
        input0 = tensors->Get(idx0);
      }
    }
    if (op->inputs() != nullptr && op->inputs()->size() > 1) {
      const int32_t idx1 = op->inputs()->Get(1);
      if (idx1 >= 0 && idx1 < tensors->size()) {
        input1 = tensors->Get(idx1);
      }
    }
    if (op->outputs() != nullptr && op->outputs()->size() > 0) {
      const int32_t idx_out = op->outputs()->Get(0);
      if (idx_out >= 0 && idx_out < tensors->size()) {
        output0 = tensors->Get(idx_out);
      }
    }

    int64_t macs_for_op = 0;

    switch (builtin_code) {
      case ::tflite::BuiltinOperator_CONV_2D:
        macs_for_op = EstimateConv2DMacs(input0, input1, output0);
        break;

      case ::tflite::BuiltinOperator_FULLY_CONNECTED:
        macs_for_op = EstimateFullyConnectedMacs(input0, input1, output0);
        break;

      case ::tflite::BuiltinOperator_MEAN:
      case ::tflite::BuiltinOperator_SUM:
        macs_for_op = EstimateReduceMacs(input0, output0);
        break;

      case ::tflite::BuiltinOperator_ADD:
      case ::tflite::BuiltinOperator_MUL:
      case ::tflite::BuiltinOperator_SOFTMAX:
      case ::tflite::BuiltinOperator_RESHAPE:
      case ::tflite::BuiltinOperator_STRIDED_SLICE:
      case ::tflite::BuiltinOperator_CONCATENATION:
      case ::tflite::BuiltinOperator_UNPACK:
      case ::tflite::BuiltinOperator_SPLIT:
      case ::tflite::BuiltinOperator_LOGISTIC:
      case ::tflite::BuiltinOperator_TANH:
      case ::tflite::BuiltinOperator_PACK:
      case ::tflite::BuiltinOperator_PAD:
      case ::tflite::BuiltinOperator_QUANTIZE:
        // Treated as 0 MACs (indexing, elementwise ops, or small cost).
        macs_for_op = 0;
        break;

      default:
        // Unknown or unsupported op: ignore in MACs estimate.
        macs_for_op = 0;
        break;
    }

    const char* op_name = ::tflite::EnumNameBuiltinOperator(builtin_code);
    if (op_name == nullptr) {
      op_name = "UNKNOWN";
    }
    std::printf("Op %u (%s): MACs=%lld\n", static_cast<unsigned>(i), op_name,
                static_cast<long long>(macs_for_op));

    if (macs_for_op > 0) {
      total_macs += static_cast<uint64_t>(macs_for_op);
    }
  }

  std::printf("Total estimated MACs (subgraph 0): %llu\n",
              static_cast<unsigned long long>(total_macs));
  return total_macs;
}

