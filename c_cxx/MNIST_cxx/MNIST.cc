#include <iostream>
#include <string>
#include <assert.h>
#include <png.h>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

#include "onnxruntime_c_api.h"
#include "providers.h"

const OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      std::cout << msg << std::endl;                         \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

template <typename T>
static void softmax(T& input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

/**
 * get_output_data from HWC format
 */
static void get_output_data(const png_byte* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t stride = h * w;
  *output_count = stride * 1;
  float* output_data = (float*)malloc(*output_count * sizeof(float));

  for (size_t i = 0; i != stride; ++i) {
    output_data[i] = input[i * 3] == 0 ? 1.0f : 0.0f;
    std::string temp = input[i * 3] == 0 ? "**" : "--";
    std::cout << temp ;
    if ((i + 1) % w == 0) { std::cout << std::endl; };
  }
  *output = output_data;
}

/**
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
static int read_png_file(const char* input_file, size_t* height, size_t* width, float** out, size_t* output_count) {
  png_image image; /* The control structure used by libpng */
  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, input_file) == 0) {
    return -1;
  }
  png_bytep buffer;
  image.format = PNG_FORMAT_BGR;
  size_t input_data_length = PNG_IMAGE_SIZE(image);
  if (input_data_length != 28 * 28 * 3) {
    std::cout << "input_data_length: " << input_data_length << std::endl;
    return -1;
  }
  buffer = (png_bytep)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer, 0 /*row_stride*/, NULL /*colormap*/) == 0) {
    return -1;
  }

  get_output_data(buffer, image.height, image.width, out, output_count);
  free(buffer);
  *width = image.width;
  *height = image.height;
  return 0;
}

static void usage() { std::cout << "usage: <model_path> <input_file> [cpu|cuda|dml]" << std::endl; }

int run_inference(OrtSession* session, const ORTCHAR_T* input_file) {
  size_t input_height;
  size_t input_width;
  float* model_input;
  size_t model_input_ele_count;

  const char* input_file_p = input_file;

  if (read_png_file(input_file_p, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    return -1;
  }

  if (input_height != 28 || input_width != 28) {
    std::cout << "please resize to image to 28x28" << std::endl;
    free(model_input);
    return -1;
  }

  OrtMemoryInfo* memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, 1, 28, 28};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);

  OrtValue* input_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
                                                           input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor));
  assert(input_tensor != NULL);
  int is_tensor;
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);
  const char* input_names[] = {"Input3"};
  const char* output_names[] = {"Plus214_Output_0"};

  std::array<float, 10> results_{};
  int result_{0};
  const int64_t output_shape[] = {1, 10};
  const size_t output_shape_len = sizeof(output_shape) / sizeof(output_shape[0]);

  OrtValue* output_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, results_.data(), results_.size()*sizeof(float), output_shape, output_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_tensor));

  ORT_ABORT_ON_ERROR(
      g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
  assert(output_tensor != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);


  softmax(results_);
  for (size_t i = 0; i < results_.size(); i++) {
   std::cout << i << " : " << results_[i] << std::endl;
  }

  result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
  std::cout << "Result : " << result_ << std::endl;

  int ret = 0;
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
  free(model_input);
  return ret;
}

void verify_input_output_count(OrtSession* session) {
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

#ifdef USE_CUDA
void enable_cuda(OrtSessionOptions* session_options) {
  ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
}
#endif

#ifdef USE_DML
void enable_dml(OrtSessionOptions* session_options) {
  ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
}
#endif

int main(int argc, char* argv[]) {
  if (argc < 3) {
    usage();
    return -1;
  }

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  ORTCHAR_T* model_path = argv[1];
  ORTCHAR_T* input_file = argv[2];
  ORTCHAR_T* execution_provider = (argc >= 4) ? argv[3] : NULL;
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  OrtSessionOptions* session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  if (execution_provider)
  {
    if (strcmp(execution_provider, ORT_TSTR("cpu")) == 0) {
      // Nothing; this is the default
    } else if (strcmp(execution_provider, ORT_TSTR("cuda")) == 0) {
    #ifdef USE_CUDA
      enable_cuda(session_options);
    #else
      puts("CUDA is not enabled in this build.");
      return -1;
    #endif
    } else if (strcmp(execution_provider, ORT_TSTR("dml")) == 0) {
    #ifdef USE_DML
      enable_dml(session_options);
    #else
      puts("DirectML is not enabled in this build.");
      return -1;
    #endif
    } else {
      usage();
      puts("Invalid execution provider option.");
      return -1;
    }
  }

  OrtSession* session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
  verify_input_output_count(session);
  int ret = run_inference(session, input_file);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);
  if (ret != 0) {
    std::cout << "fail." << std::endl;
  }
  return 0;
}
