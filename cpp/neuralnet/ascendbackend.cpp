#ifdef USE_ASCEND_BACKEND

#include "../neuralnet/nninterface.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
// ACLNN operator headers (from aclnnop directory)
// Note: inplace variants are declared in the same header as non-inplace
#include "aclnn_convolution.h"
#include "aclnn_relu.h"
#include "aclnn_add.h"
#include "aclnn_mul.h"
#include "aclnn_matmul.h"
#include "aclnn_adaptive_avg_pool2d.h"
#include "aclnn_cast.h"
#include "aclnn_fill_scalar.h"
#include "aclnn_copy.h"
#include "aclnn_cat.h"
#include "aclnn_batch_norm.h"
#include "aclnn_mish.h"
#include "aclnn_softplus.h"
#include "aclnn_tanh.h"

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/activations.h"

#include "../core/test.h"

using namespace std;

// ACLNN_SUCCESS is 0 on CANN 8.3
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

//---------------------------------------------------------------------------------
// Ascend NPU Backend for KataGo
//
// This backend provides inference acceleration on Huawei Ascend 910 Pro A NPUs
// using the Ascend Computing Language (AscendCL) and ACLNN operators.
//
// Hardware Notes:
// - No bf16 support - use fp16 or fp32 only
// - FP16 is the performance sweet spot - DaVinci Cube Unit optimized for fp16 with fp32 accumulation
// - CANN auto-fusion may need tuning - potentially disable via MS_ENABLE_ACLNN=1 if crashes occur
//
// ACLNN Two-Phase Pattern:
// Every ACLNN operator uses:
// 1. Phase 1: Call xxxGetWorkspaceSize() to get workspaceSize and aclOpExecutor*
// 2. Phase 2: Call xxx(workspace, workspaceSize, executor, stream)
//---------------------------------------------------------------------------------

// Error checking macro for AscendCL
#define ACL_CHECK(call, name) \
  do { \
    aclError err = call; \
    if(err != ACL_SUCCESS) { \
      throw StringError(string(name) + " failed with ACL error: " + to_string((int)err)); \
    } \
  } while(0)

// Error checking macro for ACLNN operators
#define ACLNN_CHECK(call, name) \
  do { \
    aclnnStatus err = call; \
    if(err != ACLNN_SUCCESS) { \
      throw StringError(string(name) + " failed with ACLNN error: " + to_string((int)err)); \
    } \
  } while(0)

// cubeMathType: 0=KEEP_DTYPE, 1=ALLOW_FP32_DOWN_PRECISION, 2=USE_FP16, 3=USE_HF32
// Ascend 910ProA Cube Unit only supports FP16, so we must use 1 to allow
// automatic FP32->FP16 downcast for computation.
static const int8_t ASCEND_CUBE_MATH_TYPE = 1;  // ALLOW_FP32_DOWN_PRECISION

//---------------------------------------------------------------------------------
// AscendCL Helper Functions
//---------------------------------------------------------------------------------

// Helper to create an aclTensor from raw device pointer
// shape is in NCHW format (batch, channels, height, width) or (batch, channels) for 2D
// CANN 8.3 aclCreateTensor signature (9 parameters):
//   aclCreateTensor(viewDims, viewDimsNum, dataType, strides, storageOffset,
//                   format, storageDims, storageDimsNum, data)
static aclTensor* createAclTensor(
  void* data,
  const vector<int64_t>& shape,
  aclDataType dtype,
  aclFormat format
) {
  if(shape.empty()) {
    return nullptr;
  }

  // Compute contiguous strides
  vector<int64_t> strides(shape.size());
  strides[shape.size() - 1] = 1;
  for(int i = (int)shape.size() - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  aclTensor* tensor = aclCreateTensor(
    shape.data(),                          // viewDims
    static_cast<uint64_t>(shape.size()),   // viewDimsNum
    dtype,                                 // dataType
    strides.data(),                        // strides
    (int64_t)0,                            // storageOffset
    format,                                // format
    shape.data(),                          // storageDims
    static_cast<uint64_t>(shape.size()),   // storageDimsNum
    data                                   // device data
  );

  return tensor;
}

// Helper to create a scalar aclScalar
static aclScalar* createAclScalar(float value, aclDataType dtype) {
  return aclCreateScalar(&value, dtype);
}

// Helper to create a float scalar with ACL_FLOAT type
static aclScalar* createFloatScalar(float value) {
  return aclCreateScalar(&value, ACL_FLOAT);
}


// Helper to destroy an aclTensor
static void destroyAclTensor(aclTensor* tensor) {
  if(tensor != nullptr) {
    aclDestroyTensor(tensor);
  }
}

// Helper to create aclIntArray
static aclIntArray* createAclIntArray(const vector<int64_t>& values) {
  return aclCreateIntArray(values.data(), static_cast<uint64_t>(values.size()));
}

//---------------------------------------------------------------------------------
// Memory Management Helpers
//---------------------------------------------------------------------------------

// Allocate device memory
static void* ascendMalloc(size_t size) {
  if(size == 0) return nullptr;
  void* ptr = nullptr;
  aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMalloc failed for size " + to_string(size) + " with error: " + to_string(ret));
  }
  return ptr;
}

// Free device memory
static void ascendFree(void* ptr) {
  if(ptr != nullptr) {
    aclrtFree(ptr);
  }
}

// Copy host to device
static void ascendCopyH2D(void* dst, const void* src, size_t size) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpy H2D failed with error: " + to_string(ret));
  }
}

// Copy device to host
static void ascendCopyD2H(void* dst, const void* src, size_t size) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpy D2H failed with error: " + to_string(ret));
  }
}

// Copy device to device
static void ascendCopyD2D(void* dst, const void* src, size_t size) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpy D2D failed with error: " + to_string(ret));
  }
}

// Allocate and copy host data to device
static void* ascendMallocAndCopy(const void* hostData, size_t size) {
  if(size == 0 || hostData == nullptr) return nullptr;
  void* devicePtr = ascendMalloc(size);
  ascendCopyH2D(devicePtr, hostData, size);
  return devicePtr;
}

// Allocate device memory with FP16 conversion: convert float[] on host to aclFloat16[], then upload
// This is the key optimization for Ascend 910ProA - native FP16 weights eliminate per-op conversion
static void* ascendMallocAndCopyFP16(const float* hostData, size_t numElements) {
  if(numElements == 0 || hostData == nullptr) return nullptr;
  // Convert to FP16 on host using CANN's conversion function
  vector<aclFloat16> fp16Data(numElements);
  for(size_t i = 0; i < numElements; i++) {
    fp16Data[i] = aclFloatToFloat16(hostData[i]);
  }
  size_t fp16Bytes = numElements * sizeof(aclFloat16);
  void* devicePtr = ascendMalloc(fp16Bytes);
  ascendCopyH2D(devicePtr, fp16Data.data(), fp16Bytes);
  return devicePtr;
}

// Overload for vector<float>
static void* ascendMallocAndCopyFP16(const vector<float>& hostData) {
  return ascendMallocAndCopyFP16(hostData.data(), hostData.size());
}


//---------------------------------------------------------------------------------
// LoadedModel - simple wrapper around ModelDesc
//---------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//---------------------------------------------------------------------------------
// Global Initialization/Cleanup
//---------------------------------------------------------------------------------

static bool g_aclInitialized = false;

void NeuralNet::globalInitialize() {
  if(!g_aclInitialized) {
    aclError ret = aclInit(nullptr);
    if(ret != ACL_SUCCESS) {
      throw StringError("aclInit failed with error code: " + to_string(ret));
    }
    g_aclInitialized = true;
  }
}

void NeuralNet::globalCleanup() {
  if(g_aclInitialized) {
    aclFinalize();
    g_aclInitialized = false;
  }
}

void NeuralNet::printDevices() {
  uint32_t deviceCount = 0;
  aclError ret = aclrtGetDeviceCount(&deviceCount);
  if(ret != ACL_SUCCESS) {
    cout << "Failed to get Ascend NPU device count" << endl;
    return;
  }

  cout << "Found " << deviceCount << " Ascend NPU device(s)" << endl;
  for(uint32_t i = 0; i < deviceCount; i++) {
    // Note: aclrtGetSocName requires a device to be set first
    // For now, just print the index
    cout << "  Ascend NPU device " << i << endl;
  }
}

//---------------------------------------------------------------------------------
// Forward declarations
//---------------------------------------------------------------------------------

struct Model;
struct ScratchBuffers;
struct Buffers;

//---------------------------------------------------------------------------------
// ComputeContext - cross-thread NPU state
//---------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  enabled_t useNHWCMode;
  vector<int> gpuIdxs;

  std::mutex cachedModelsMutex;
  std::map<std::string, std::shared_ptr<const Model>> cachedModels;
  std::map<std::string, int> cachedModelsRefCount;

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  ComputeContext(int nnX, int nnY, enabled_t fp16, enabled_t nhwc, const vector<int>& gpus)
    : nnXLen(nnX),
      nnYLen(nnY),
      useFP16Mode(fp16),
      useNHWCMode(nhwc),
      gpuIdxs(gpus)
  {}

  ~ComputeContext() {
    assert(cachedModels.size() == 0);
  }
};

//---------------------------------------------------------------------------------
// ComputeHandle - per-thread handle
//---------------------------------------------------------------------------------

struct ComputeHandle {
  int deviceIdx;
  aclrtStream stream;

  const Model* model;
  ScratchBuffers* scratch;
  Buffers* buffers;

  bool usingFP16;
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;
  bool inputsUseNHWC;

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  ComputeHandle(int device, bool fp16, int nnX, int nnY, bool exactLen, bool nhwc)
    : deviceIdx(device),
      stream(nullptr),
      model(nullptr),
      scratch(nullptr),
      buffers(nullptr),
      usingFP16(fp16),
      nnXLen(nnX),
      nnYLen(nnY),
      requireExactNNLen(exactLen),
      inputsUseNHWC(nhwc)
  {}

  ~ComputeHandle() {
    // Set device context first since destructor may run on a different thread
    // CANN's device binding is thread-local
    aclrtSetDevice(deviceIdx);
    if(stream != nullptr) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceIdx);
  }
};

//---------------------------------------------------------------------------------
// InputBuffers - host-side buffers
//---------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singleInputMetaElts;
  size_t singleInputMetaBytes;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyPassResultBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t userInputMetaBufferBytes;
  size_t policyPassResultBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer;
  float* userInputGlobalBuffer;
  float* userInputMetaBuffer;

  float* policyPassResults;
  float* policyResults;
  float* valueResults;
  float* scoreValueResults;
  float* ownershipResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnX, int nnY) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnX * nnY;
    singleInputBytes = singleInputElts * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = singleInputGlobalElts * sizeof(float);
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    singleInputMetaBytes = singleInputMetaElts * sizeof(float);

    singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    singlePolicyPassResultBytes = singlePolicyPassResultElts * sizeof(float);
    singlePolicyResultElts = (size_t)m.numPolicyChannels * nnX * nnY;
    singlePolicyResultBytes = singlePolicyResultElts * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = singleValueResultElts * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = singleScoreValueResultElts * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnX * nnY;
    singleOwnershipResultBytes = singleOwnershipResultElts * sizeof(float);

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSz * nnX * nnY * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSz * sizeof(float);
    userInputMetaBufferBytes = (size_t)m.numInputMetaChannels * maxBatchSz * sizeof(float);
    policyPassResultBufferBytes = (size_t)maxBatchSz * m.numPolicyChannels * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSz * m.numPolicyChannels * nnX * nnY * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSz * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSz * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSz * nnX * nnY * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[singleInputElts * maxBatchSz];
    userInputGlobalBuffer = new float[singleInputGlobalElts * maxBatchSz];
    if(m.numInputMetaChannels > 0) {
      userInputMetaBuffer = new float[singleInputMetaElts * maxBatchSz];
    } else {
      userInputMetaBuffer = nullptr;
    }

    policyPassResults = new float[singlePolicyPassResultElts * maxBatchSz];
    policyResults = new float[singlePolicyResultElts * maxBatchSz];
    valueResults = new float[singleValueResultElts * maxBatchSz];
    scoreValueResults = new float[singleScoreValueResultElts * maxBatchSz];
    ownershipResults = new float[singleOwnershipResultElts * maxBatchSz];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    if(userInputMetaBuffer != nullptr) {
      delete[] userInputMetaBuffer;
    }
    delete[] policyPassResults;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//---------------------------------------------------------------------------------
// ScratchBuffers - workspace allocator
//---------------------------------------------------------------------------------

struct ScratchBuffers {
  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const bool useFP16;

  // Pre-allocated workspace for ACLNN operations
  void* workspaceBuf;
  size_t workspaceBytes;

  // Track allocated buffers for cleanup
  vector<void*> allocatedBuffers;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(int maxBatchSz, int nnX, int nnY, bool fp16, size_t maxWorkspaceNeeded)
    : batchXYFloatBytes((size_t)maxBatchSz * nnX * nnY * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSz * sizeof(float)),
      batchXYBytes((size_t)maxBatchSz * nnX * nnY * (fp16 ? sizeof(aclFloat16) : sizeof(float))),
      batchBytes((size_t)maxBatchSz * (fp16 ? sizeof(aclFloat16) : sizeof(float))),
      maxBatchSize(maxBatchSz),
      nnXLen(nnX),
      nnYLen(nnY),
      useFP16(fp16),
      workspaceBuf(nullptr),
      workspaceBytes(0)
  {
    // Pre-allocate workspace
    workspaceBytes = maxWorkspaceNeeded;
    if(workspaceBytes > 0) {
      workspaceBuf = ascendMalloc(workspaceBytes);
    }
  }

  ~ScratchBuffers() {
    // Free any allocated buffers
    for(void* buf : allocatedBuffers) {
      ascendFree(buf);
    }
    if(workspaceBuf != nullptr) {
      ascendFree(workspaceBuf);
    }
  }

  // Allocate a buffer of the given size and track it for cleanup
  void* allocate(size_t size) {
    void* buf = ascendMalloc(size);
    allocatedBuffers.push_back(buf);
    return buf;
  }

  // Release tracking for a buffer (does not free immediately, just removes from tracking)
  void release(void* buf) {
    // Find and remove from tracking (buffer will be freed in destructor)
    for(auto it = allocatedBuffers.begin(); it != allocatedBuffers.end(); ++it) {
      if(*it == buf) {
        allocatedBuffers.erase(it);
        ascendFree(buf);
        return;
      }
    }
  }

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }
};

//---------------------------------------------------------------------------------
// Buffers - device-side buffers
//---------------------------------------------------------------------------------

struct Buffers {
  // Input buffers (device)
  void* inputBuf;           // Spatial input (NCHW)
  void* inputGlobalBuf;     // Global features (NC)
  void* inputMetaBuf;       // Meta features (NC, optional)

  // For FP16 mode, we also need float versions for initial copy
  void* inputBufFloat;
  void* inputGlobalBufFloat;
  void* inputMetaBufFloat;

  // Output buffers (device, always float32 for final output)
  float* policyPassBuf;
  float* policyBuf;
  float* valueBuf;
  float* scoreValueBuf;
  void* ownershipBuf;

  // Workspace
  void* workspaceBuf;
  size_t workspaceBytes;

  // Size tracking
  size_t inputBufBytes;
  size_t inputGlobalBufBytes;
  size_t inputMetaBufBytes;
  size_t inputBufBytesFloat;
  size_t inputGlobalBufBytesFloat;
  size_t inputMetaBufBytesFloat;
  size_t policyPassBufBytes;
  size_t policyBufBytes;
  size_t valueBufBytes;
  size_t scoreValueBufBytes;
  size_t ownershipBufBytes;

  Buffers(const ModelDesc& m, int maxBatchSize, int nnXLen, int nnYLen, bool useFP16, size_t extraWorkspace)
    : inputBuf(nullptr),
      inputGlobalBuf(nullptr),
      inputMetaBuf(nullptr),
      inputBufFloat(nullptr),
      inputGlobalBufFloat(nullptr),
      inputMetaBufFloat(nullptr),
      policyPassBuf(nullptr),
      policyBuf(nullptr),
      valueBuf(nullptr),
      scoreValueBuf(nullptr),
      ownershipBuf(nullptr),
      workspaceBuf(nullptr),
      workspaceBytes(0)
  {
    size_t eltSize = useFP16 ? sizeof(aclFloat16) : sizeof(float);

    inputBufBytes = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * eltSize;
    inputGlobalBufBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * eltSize;
    inputMetaBufBytes = (size_t)m.numInputMetaChannels * maxBatchSize * eltSize;

    inputBufBytesFloat = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * sizeof(float);
    inputGlobalBufBytesFloat = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    inputMetaBufBytesFloat = (size_t)m.numInputMetaChannels * maxBatchSize * sizeof(float);

    policyPassBufBytes = (size_t)maxBatchSize * m.numPolicyChannels * sizeof(float);
    policyBufBytes = (size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen * sizeof(float);
    valueBufBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueBufBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipBufBytes = (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels * sizeof(float);

    // Allocate input buffers
    inputBuf = ascendMalloc(inputBufBytes);
    inputGlobalBuf = ascendMalloc(inputGlobalBufBytes);
    if(m.numInputMetaChannels > 0) {
      inputMetaBuf = ascendMalloc(inputMetaBufBytes);
    }

    // For FP16 mode, allocate float buffers for initial host copy
    if(useFP16) {
      inputBufFloat = ascendMalloc(inputBufBytesFloat);
      inputGlobalBufFloat = ascendMalloc(inputGlobalBufBytesFloat);
      if(m.numInputMetaChannels > 0) {
        inputMetaBufFloat = ascendMalloc(inputMetaBufBytesFloat);
      }
    }

    // Allocate output buffers (always float32)
    policyPassBuf = (float*)ascendMalloc(policyPassBufBytes);
    policyBuf = (float*)ascendMalloc(policyBufBytes);
    valueBuf = (float*)ascendMalloc(valueBufBytes);
    scoreValueBuf = (float*)ascendMalloc(scoreValueBufBytes);
    ownershipBuf = ascendMalloc(ownershipBufBytes);

    // Allocate workspace
    workspaceBytes = extraWorkspace;
    if(workspaceBytes > 0) {
      workspaceBuf = ascendMalloc(workspaceBytes);
    }
  }

  ~Buffers() {
    ascendFree(inputBuf);
    ascendFree(inputGlobalBuf);
    ascendFree(inputMetaBuf);
    ascendFree(inputBufFloat);
    ascendFree(inputGlobalBufFloat);
    ascendFree(inputMetaBufFloat);
    ascendFree(policyPassBuf);
    ascendFree(policyBuf);
    ascendFree(valueBuf);
    ascendFree(scoreValueBuf);
    ascendFree(ownershipBuf);
    ascendFree(workspaceBuf);
  }

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
};

//---------------------------------------------------------------------------------
// Basic Layer Implementations
//---------------------------------------------------------------------------------

// ConvLayer - convolution layer using ACLNN
struct ConvLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  const int convYSize;
  const int convXSize;
  const int dilationY;
  const int dilationX;

  void* filterBuf;              // Device memory for weights (NCHW: outC, inC, H, W)
  aclDataType dtype;
  bool useFP16;
  int8_t cubeMathType;          // 0=KEEP_DTYPE (native FP16), 1=ALLOW_FP32_DOWN_PRECISION

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc* desc, bool useFP16_)
    : name(desc->name),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      convYSize(desc->convYSize),
      convXSize(desc->convXSize),
      dilationY(desc->dilationY),
      dilationX(desc->dilationX),
      useFP16(useFP16_)
  {
    // Allocate and copy weights to device with native FP16 conversion
    // KataGo weights are in (outC, inC, H, W) format which is NCHW-compatible
    if(useFP16) {
      filterBuf = ascendMallocAndCopyFP16(desc->weights.data(), desc->weights.size());
      dtype = ACL_FLOAT16;
      cubeMathType = 0;  // KEEP_DTYPE - weights are already native FP16
    } else {
      size_t weightBytes = desc->weights.size() * sizeof(float);
      filterBuf = ascendMallocAndCopy(desc->weights.data(), weightBytes);
      dtype = ACL_FLOAT;
      cubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION - let CANN convert FP32->FP16
    }
  }

  ~ConvLayer() {
    ascendFree(filterBuf);
  }

  size_t requiredWorkspaceBytes(int batchSize, int nnXLen, int nnYLen, aclrtStream stream) const {
    // Query ACLNN for workspace size
    // Create dummy tensors to query
    vector<int64_t> inputShape = {batchSize, inChannels, nnYLen, nnXLen};
    vector<int64_t> outputShape = {batchSize, outChannels, nnYLen, nnXLen};
    vector<int64_t> weightShape = {outChannels, inChannels, convYSize, convXSize};

    // Create tensors (with nullptr data for size query)
    aclTensor* inputTensor = createAclTensor(nullptr, inputShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* weightTensor = createAclTensor(nullptr, weightShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* outputTensor = createAclTensor(nullptr, outputShape, dtype, ACL_FORMAT_NCHW);

    // Compute padding
    int paddingY = (convYSize / 2) * dilationY;
    int paddingX = (convXSize / 2) * dilationX;

    // Create arrays for convolution parameters
    aclIntArray* stridesArr = createAclIntArray({1, 1});
    aclIntArray* paddingsArr = createAclIntArray({paddingY, paddingX});
    aclIntArray* dilationsArr = createAclIntArray({dilationY, dilationX});
    aclIntArray* outputPaddingArr = createAclIntArray({0, 0});

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // CANN 8.3 signature:
    // aclnnConvolutionGetWorkspaceSize(input, weight, bias, stride, padding, dilation,
    //                                   transposed, outputPadding, groups, output,
    //                                   cubeMathType, workspaceSize, executor)
    aclnnStatus status = aclnnConvolutionGetWorkspaceSize(
      inputTensor,
      weightTensor,
      nullptr,        // bias
      stridesArr,
      paddingsArr,
      dilationsArr,
      false,          // transposed
      outputPaddingArr,
      (int64_t)1,     // groups
      outputTensor,
      cubeMathType,      // cubeMathType: 0 for native FP16, 1 for FP32->FP16 conversion
      &workspaceSize,
      &executor
    );

    // Cleanup
    destroyAclTensor(inputTensor);
    destroyAclTensor(weightTensor);
    destroyAclTensor(outputTensor);
    aclDestroyIntArray(stridesArr);
    aclDestroyIntArray(paddingsArr);
    aclDestroyIntArray(dilationsArr);
    aclDestroyIntArray(outputPaddingArr);

    if(status != ACLNN_SUCCESS) {
      // Return a conservative estimate
      return 1024 * 1024 * 16; // 16 MB fallback
    }

    (void)stream;
    return workspaceSize;
  }

  void apply(
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Create tensors
    vector<int64_t> inputShape = {batchSize, inChannels, nnYLen, nnXLen};
    vector<int64_t> outputShape = {batchSize, outChannels, nnYLen, nnXLen};
    vector<int64_t> weightShape = {outChannels, inChannels, convYSize, convXSize};

    aclTensor* inputTensor = createAclTensor(inputBuf, inputShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* weightTensor = createAclTensor(filterBuf, weightShape, dtype, ACL_FORMAT_NCHW);

    // For accumulate mode, ACLNN convolution doesn't support beta parameter directly
    // We need to handle this differently
    aclTensor* outputTensor;
    void* actualOutputBuf = outputBuf;

    if(accumulate) {
      // For accumulate mode, we need to:
      // 1. Compute convolution into a temp buffer
      // 2. Add to output buffer
      // For simplicity, we'll just not support accumulate for now
      // and rely on the caller to handle it
      outputTensor = createAclTensor(actualOutputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
    } else {
      outputTensor = createAclTensor(actualOutputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
    }

    // Compute padding
    int paddingY = (convYSize / 2) * dilationY;
    int paddingX = (convXSize / 2) * dilationX;

    // Create arrays for convolution parameters
    aclIntArray* stridesArr = createAclIntArray({1, 1});
    aclIntArray* paddingsArr = createAclIntArray({paddingY, paddingX});
    aclIntArray* dilationsArr = createAclIntArray({dilationY, dilationX});
    aclIntArray* outputPaddingArr = createAclIntArray({0, 0});

    // Phase 1: Get workspace size
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus status = aclnnConvolutionGetWorkspaceSize(
      inputTensor,
      weightTensor,
      nullptr,        // bias
      stridesArr,
      paddingsArr,
      dilationsArr,
      false,          // transposed
      outputPaddingArr,
      (int64_t)1,     // groups
      outputTensor,
      cubeMathType,      // cubeMathType: 0 for native FP16, 1 for FP32->FP16 conversion
      &wsSize,
      &executor
    );

    if(status != ACLNN_SUCCESS) {
      destroyAclTensor(inputTensor);
      destroyAclTensor(weightTensor);
      destroyAclTensor(outputTensor);
      aclDestroyIntArray(stridesArr);
      aclDestroyIntArray(paddingsArr);
      aclDestroyIntArray(dilationsArr);
      aclDestroyIntArray(outputPaddingArr);
      throw StringError("aclnnConvolutionGetWorkspaceSize failed for layer " + name + " with error: " + to_string(status));
    }

    // Phase 2: Execute
    status = aclnnConvolution(workspaceBuf, wsSize, executor, stream);

    // Cleanup
    destroyAclTensor(inputTensor);
    destroyAclTensor(weightTensor);
    destroyAclTensor(outputTensor);
    aclDestroyIntArray(stridesArr);
    aclDestroyIntArray(paddingsArr);
    aclDestroyIntArray(dilationsArr);
    aclDestroyIntArray(outputPaddingArr);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnConvolution failed for layer " + name + " with error: " + to_string(status));
    }

    // Handle accumulate mode with separate add if needed
    if(accumulate) {
      // TODO: Implement accumulate using aclnnAdd
      // For now, this is not used in the initial implementation
    }

    (void)workspaceBytes;
  }
};

// BatchNormLayer - merged scale+bias with optional activation
struct BatchNormLayer {
  const string name;
  const int numChannels;
  const int activation;  // ACTIVATION_IDENTITY, RELU, or MISH

  void* mergedScaleBuf;  // Device memory
  void* mergedBiasBuf;   // Device memory
  bool useFP16;
  int nnXLen;
  int nnYLen;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(const BatchNormLayerDesc* desc, const ActivationLayerDesc* actDesc, int nnX, int nnY, bool useFP16_)
    : name(desc->name),
      numChannels(desc->numChannels),
      activation(actDesc ? actDesc->activation : ACTIVATION_IDENTITY),
      useFP16(useFP16_),
      nnXLen(nnX),
      nnYLen(nnY)
  {
    // Allocate and copy merged scale and bias with native FP16 conversion
    if(useFP16) {
      mergedScaleBuf = ascendMallocAndCopyFP16(desc->mergedScale);
      mergedBiasBuf = ascendMallocAndCopyFP16(desc->mergedBias);
    } else {
      size_t scaleBytes = desc->mergedScale.size() * sizeof(float);
      size_t biasBytes = desc->mergedBias.size() * sizeof(float);
      mergedScaleBuf = ascendMallocAndCopy(desc->mergedScale.data(), scaleBytes);
      mergedBiasBuf = ascendMallocAndCopy(desc->mergedBias.data(), biasBytes);
    }
  }

  ~BatchNormLayer() {
    ascendFree(mergedScaleBuf);
    ascendFree(mergedBiasBuf);
  }

  void apply(
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    const void* maskBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // BatchNorm: output = input * scale + bias, then apply activation
    // scale and bias are (numChannels,) shaped, need to broadcast to (N, C, H, W)

    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    // Step 1: Multiply input by scale
    // input: (N, C, H, W), scale: (1, C, 1, 1) -> output: (N, C, H, W)
    {
      vector<int64_t> inputShape = {batchSize, numChannels, nnYLen, nnXLen};
      vector<int64_t> scaleShape = {1, numChannels, 1, 1};  // Broadcast to NCHW

      aclTensor* inputTensor = createAclTensor(inputBuf, inputShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* scaleTensor = createAclTensor(mergedScaleBuf, scaleShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* outputTensor = createAclTensor(outputBuf, inputShape, dtype, ACL_FORMAT_NCHW);

      // Two-phase pattern for mul
      uint64_t mulWsSize = 0;
      aclOpExecutor* mulExecutor = nullptr;

      aclnnStatus status = aclnnMulGetWorkspaceSize(inputTensor, scaleTensor, outputTensor, &mulWsSize, &mulExecutor);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(inputTensor);
        destroyAclTensor(scaleTensor);
        destroyAclTensor(outputTensor);
        throw StringError("aclnnMulGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnMul(workspaceBuf, mulWsSize, mulExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(inputTensor);
        destroyAclTensor(scaleTensor);
        destroyAclTensor(outputTensor);
        throw StringError("aclnnMul failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      destroyAclTensor(inputTensor);
      destroyAclTensor(scaleTensor);
      destroyAclTensor(outputTensor);
    }

    // Step 2: Add bias
    {
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};
      vector<int64_t> biasShape = {1, numChannels, 1, 1};  // Broadcast to NCHW

      aclTensor* outputTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* biasTensor = createAclTensor(mergedBiasBuf, biasShape, dtype, ACL_FORMAT_NCHW);
      aclScalar* alpha = createFloatScalar(1.0f);

      uint64_t addWsSize = 0;
      aclOpExecutor* addExecutor = nullptr;

      aclnnStatus status = aclnnAddGetWorkspaceSize(outputTensor, biasTensor, alpha, outputTensor, &addWsSize, &addExecutor);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        destroyAclTensor(biasTensor);
        aclDestroyScalar(alpha);
        throw StringError("aclnnAddGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        destroyAclTensor(biasTensor);
        aclDestroyScalar(alpha);
        throw StringError("aclnnAdd failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      destroyAclTensor(outputTensor);
      destroyAclTensor(biasTensor);
      aclDestroyScalar(alpha);
    }

    // Step 3: Apply activation (relu if needed)
    if(activation == ACTIVATION_RELU) {
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};

      aclTensor* outputTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);

      uint64_t reluWsSize = 0;
      aclOpExecutor* reluExecutor = nullptr;

      aclnnStatus status = aclnnInplaceReluGetWorkspaceSize(outputTensor, &reluWsSize, &reluExecutor);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        throw StringError("aclnnInplaceReluGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnInplaceRelu(workspaceBuf, reluWsSize, reluExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        throw StringError("aclnnInplaceRelu failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      destroyAclTensor(outputTensor);
    }
    // MISH activation using native ACLNN operator
    else if(activation == ACTIVATION_MISH || activation == ACTIVATION_MISH_SCALE8) {
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};

      aclTensor* outputTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);

      uint64_t mishWsSize = 0;
      aclOpExecutor* mishExecutor = nullptr;

      aclnnStatus status = aclnnInplaceMishGetWorkspaceSize(outputTensor, &mishWsSize, &mishExecutor);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        throw StringError("aclnnInplaceMishGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnInplaceMish(workspaceBuf, mishWsSize, mishExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        throw StringError("aclnnInplaceMish failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      destroyAclTensor(outputTensor);

      // For ACTIVATION_MISH_SCALE8, scale the output by 8.0
      // mish_scale8(x) = 8.0 * mish(x)
      if(activation == ACTIVATION_MISH_SCALE8) {
        aclTensor* scaledTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
        aclScalar* scaleScalar = createFloatScalar(8.0f);

        uint64_t mulWsSize = 0;
        aclOpExecutor* mulExecutor = nullptr;

        status = aclnnInplaceMulsGetWorkspaceSize(scaledTensor, scaleScalar, &mulWsSize, &mulExecutor);
        if(status != ACLNN_SUCCESS) {
          destroyAclTensor(scaledTensor);
          aclDestroyScalar(scaleScalar);
          throw StringError("aclnnInplaceMulsGetWorkspaceSize failed for MISH_SCALE8 " + name + " with error: " + to_string(status));
        }

        status = aclnnInplaceMuls(workspaceBuf, mulWsSize, mulExecutor, stream);
        if(status != ACLNN_SUCCESS) {
          destroyAclTensor(scaledTensor);
          aclDestroyScalar(scaleScalar);
          throw StringError("aclnnInplaceMuls failed for MISH_SCALE8 " + name + " with error: " + to_string(status));
        }

        destroyAclTensor(scaledTensor);
        aclDestroyScalar(scaleScalar);
      }
    }

    // Step 4: Apply mask if provided
    // The mask zeros out the padded regions
    if(maskBuf != nullptr) {
      // output = output * mask (elementwise)
      // mask is (N, H, W) shaped, need to broadcast
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};
      vector<int64_t> maskShape = {batchSize, 1, nnYLen, nnXLen};  // N, 1, H, W for broadcasting

      aclTensor* outputTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* maskTensor = createAclTensor(const_cast<void*>(maskBuf), maskShape, dtype, ACL_FORMAT_NCHW);

      uint64_t maskWsSize = 0;
      aclOpExecutor* maskExecutor = nullptr;

      aclnnStatus status = aclnnInplaceMulGetWorkspaceSize(outputTensor, maskTensor, &maskWsSize, &maskExecutor);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        destroyAclTensor(maskTensor);
        throw StringError("aclnnInplaceMulGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnInplaceMul(workspaceBuf, maskWsSize, maskExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        destroyAclTensor(outputTensor);
        destroyAclTensor(maskTensor);
        throw StringError("aclnnInplaceMul failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      destroyAclTensor(outputTensor);
      destroyAclTensor(maskTensor);
    }

    (void)workspaceBytes;
  }
};

// MatMulLayer - matrix multiplication
struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;

  void* matBuf;  // Device memory for weights
  bool useFP16;
  int8_t cubeMathType;  // 0=KEEP_DTYPE (native FP16), 1=ALLOW_FP32_DOWN_PRECISION

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(const MatMulLayerDesc* desc, bool useFP16_)
    : name(desc->name),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      useFP16(useFP16_)
  {
    // Weights are in (inC, outC) format
    // Allocate and copy with native FP16 conversion for optimal performance
    if(useFP16) {
      matBuf = ascendMallocAndCopyFP16(desc->weights.data(), desc->weights.size());
      cubeMathType = 0;  // KEEP_DTYPE - weights are already native FP16
    } else {
      size_t weightBytes = desc->weights.size() * sizeof(float);
      matBuf = ascendMallocAndCopy(desc->weights.data(), weightBytes);
      cubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION - let CANN convert FP32->FP16
    }
  }

  ~MatMulLayer() {
    ascendFree(matBuf);
  }

  void apply(
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Matrix multiplication: output = input @ weights^T
    // input: (batch, inC), weights: (inC, outC), output: (batch, outC)
    // Note: KataGo stores weights in (inC, outC) format
    // For matmul with input (N, inC) @ weights (inC, outC) -> output (N, outC)

    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    // Create input tensor: (N, inC)
    vector<int64_t> inputShape = {batchSize, inChannels};
    aclTensor* inputTensor = createAclTensor(inputBuf, inputShape, dtype, ACL_FORMAT_ND);

    // Create weight tensor: (inC, outC)
    vector<int64_t> weightShape = {inChannels, outChannels};
    aclTensor* weightTensor = createAclTensor(matBuf, weightShape, dtype, ACL_FORMAT_ND);

    // Create output tensor: (N, outC)
    vector<int64_t> outputShape = {batchSize, outChannels};
    aclTensor* outputTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_ND);

    // Phase 1: Get workspace size
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus status = aclnnMatmulGetWorkspaceSize(
      inputTensor,
      weightTensor,
      outputTensor,
      cubeMathType,       // cubeMathType: 0 for native FP16, 1 for FP32->FP16 conversion
      &wsSize,
      &executor
    );

    if(status != ACLNN_SUCCESS) {
      destroyAclTensor(inputTensor);
      destroyAclTensor(weightTensor);
      destroyAclTensor(outputTensor);
      throw StringError("aclnnMatmulGetWorkspaceSize failed for MatMul " + name + " with error: " + to_string(status));
    }

    // Verify workspace is sufficient
    if(wsSize > workspaceBytes) {
      destroyAclTensor(inputTensor);
      destroyAclTensor(weightTensor);
      destroyAclTensor(outputTensor);
      throw StringError("MatMul " + name + " requires more workspace: " + to_string(wsSize) + " > " + to_string(workspaceBytes));
    }

    // Phase 2: Execute
    status = aclnnMatmul(workspaceBuf, wsSize, executor, stream);

    // Cleanup
    destroyAclTensor(inputTensor);
    destroyAclTensor(weightTensor);
    destroyAclTensor(outputTensor);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnMatmul failed for MatMul " + name + " with error: " + to_string(status));
    }

    (void)workspaceBytes;
  }
};

// MatBiasLayer - bias addition
struct MatBiasLayer {
  const string name;
  const int numChannels;

  void* biasBuf;  // Device memory for bias
  bool useFP16;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(const MatBiasLayerDesc* desc, bool useFP16_)
    : name(desc->name),
      numChannels(desc->numChannels),
      useFP16(useFP16_)
  {
    // Allocate and copy bias with native FP16 conversion
    if(useFP16) {
      biasBuf = ascendMallocAndCopyFP16(desc->weights.data(), desc->weights.size());
    } else {
      size_t biasBytes = desc->weights.size() * sizeof(float);
      biasBuf = ascendMallocAndCopy(desc->weights.data(), biasBytes);
    }
  }

  ~MatBiasLayer() {
    ascendFree(biasBuf);
  }

  void apply(
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Add bias: output = input + bias
    // input: (batch, numChannels), bias: (numChannels,), output: (batch, numChannels)
    // Bias needs to be broadcast across the batch dimension

    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    // Create input tensor: (N, C)
    vector<int64_t> inputShape = {batchSize, numChannels};
    aclTensor* inputTensor = createAclTensor(inputBuf, inputShape, dtype, ACL_FORMAT_ND);

    // Create bias tensor: (1, C) - will broadcast across batch
    vector<int64_t> biasShape = {1, numChannels};
    aclTensor* biasTensor = createAclTensor(biasBuf, biasShape, dtype, ACL_FORMAT_ND);

    // Create output tensor: (N, C)
    vector<int64_t> outputShape = {batchSize, numChannels};
    aclTensor* outputTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_ND);

    // Create scalar for alpha (out = self + alpha * other)
    aclScalar* alpha = createFloatScalar(1.0f);

    // Phase 1: Get workspace size
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus status = aclnnAddGetWorkspaceSize(inputTensor, biasTensor, alpha, outputTensor, &wsSize, &executor);

    if(status != ACLNN_SUCCESS) {
      destroyAclTensor(inputTensor);
      destroyAclTensor(biasTensor);
      destroyAclTensor(outputTensor);
      aclDestroyScalar(alpha);
      throw StringError("aclnnAddGetWorkspaceSize failed for MatBias " + name + " with error: " + to_string(status));
    }

    // Verify workspace is sufficient
    if(wsSize > workspaceBytes) {
      destroyAclTensor(inputTensor);
      destroyAclTensor(biasTensor);
      destroyAclTensor(outputTensor);
      aclDestroyScalar(alpha);
      throw StringError("MatBias " + name + " requires more workspace: " + to_string(wsSize) + " > " + to_string(workspaceBytes));
    }

    // Phase 2: Execute
    status = aclnnAdd(workspaceBuf, wsSize, executor, stream);

    // Cleanup
    destroyAclTensor(inputTensor);
    destroyAclTensor(biasTensor);
    destroyAclTensor(outputTensor);
    aclDestroyScalar(alpha);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAdd failed for MatBias " + name + " with error: " + to_string(status));
    }

    (void)workspaceBytes;
  }
};

//---------------------------------------------------------------------------------
// Composite Layer Implementations
//---------------------------------------------------------------------------------

// NormActConv - BatchNorm + Activation + Conv fused pattern
struct NormActConv {
  const string name;
  unique_ptr<BatchNormLayer> bnLayer;
  unique_ptr<ConvLayer> convLayer;
  int numChannels;

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

  NormActConv(
    const BatchNormLayerDesc* bnDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnXLen,
    int nnYLen,
    bool useFP16
  ) : name(bnDesc->name + "_" + convDesc->name), numChannels(convDesc->outChannels)
  {
    bnLayer = make_unique<BatchNormLayer>(bnDesc, actDesc, nnXLen, nnYLen, useFP16);
    convLayer = make_unique<ConvLayer>(convDesc, useFP16);
  }

  size_t requiredWorkspaceBytes(int batchSize, int nnXLen, int nnYLen, aclrtStream stream) const {
    return convLayer->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream);
  }

  void apply(
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const void* maskBuf,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Apply BN + activation in-place on input
    bnLayer->apply(stream, batchSize, inputBuf, maskBuf, inputBuf, workspaceBuf, workspaceBytes);

    // Then apply convolution
    convLayer->apply(stream, batchSize, nnXLen, nnYLen, false, inputBuf, outputBuf, workspaceBuf, workspaceBytes);
  }
};

// ResidualBlock - Two NormActConvs with residual addition
struct ResidualBlock {
  const string name;
  unique_ptr<NormActConv> preNormActConv;
  unique_ptr<NormActConv> midNormActConv;
  unique_ptr<ConvLayer> finalConv;
  int numChannels;
  bool useFP16;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    const ResidualBlockDesc* desc,
    int nnXLen,
    int nnYLen,
    bool useFP16_
  ) : name(desc->name), numChannels(desc->finalConv.outChannels), useFP16(useFP16_)
  {
    preNormActConv = make_unique<NormActConv>(
      &desc->preBN, &desc->preActivation, &desc->regularConv, nnXLen, nnYLen, useFP16_);
    midNormActConv = make_unique<NormActConv>(
      &desc->midBN, &desc->midActivation, &desc->finalConv, nnXLen, nnYLen, useFP16_);
  }

  size_t requiredWorkspaceBytes(int batchSize, int nnXLen, int nnYLen, aclrtStream stream) const {
    size_t bytes = 0;
    bytes = max(bytes, preNormActConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    bytes = max(bytes, midNormActConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    return bytes;
  }

  void apply(
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const void* maskBuf,
    void* inputBuf,
    void* outputBuf,
    void* scratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Copy input to scratch as residual backup
    // We need to preserve the original input for the residual connection
    size_t trunkBytes = numChannels * batchSize * nnXLen * nnYLen * sizeof(float);
    ascendCopyD2D(scratchBuf, inputBuf, trunkBytes);  // scratch = input (save for residual)

    // Apply pre NormActConv: input -> input (in-place for BN part, then conv to output)
    // But we need a separate buffer, so: input -> output
    preNormActConv->apply(stream, batchSize, nnXLen, nnYLen, maskBuf, inputBuf, outputBuf, workspaceBuf, workspaceBytes);

    // Apply mid NormActConv: output -> output (in-place BN then conv)
    midNormActConv->apply(stream, batchSize, nnXLen, nnYLen, maskBuf, outputBuf, outputBuf, workspaceBuf, workspaceBytes);

    // Add residual: output + scratch -> output
    // This is the key step - add the saved residual to the transformed output
    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    // Create tensors for addition
    vector<int64_t> addShape = {batchSize, numChannels, nnYLen, nnXLen};
    aclTensor* outputTensor = createAclTensor(outputBuf, addShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* residualTensor = createAclTensor(scratchBuf, addShape, dtype, ACL_FORMAT_NCHW);

    // Create output tensor for the result
    vector<int64_t> resultShape = {batchSize, numChannels, nnYLen, nnXLen};
    aclTensor* resultTensor = createAclTensor(outputBuf, resultShape, dtype, ACL_FORMAT_NCHW);

    // Create scalar for alpha
    aclScalar* alpha = createFloatScalar(1.0f);

    // Phase 1: Get workspace size for add
    uint64_t addWsSize = 0;
    aclOpExecutor* addExecutor = nullptr;

    aclnnStatus status = aclnnAddGetWorkspaceSize(outputTensor, residualTensor, alpha, resultTensor, &addWsSize, &addExecutor);

    if(status != ACLNN_SUCCESS) {
      destroyAclTensor(outputTensor);
      destroyAclTensor(residualTensor);
      destroyAclTensor(resultTensor);
      aclDestroyScalar(alpha);
      throw StringError("aclnnAddGetWorkspaceSize failed for ResidualBlock " + name + " with error: " + to_string(status));
    }

    // Phase 2: Execute add
    status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);

    // Cleanup
    destroyAclTensor(outputTensor);
    destroyAclTensor(residualTensor);
    destroyAclTensor(resultTensor);
    aclDestroyScalar(alpha);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAdd failed for ResidualBlock " + name + " with error: " + to_string(status));
    }

    (void)workspaceBytes;
  }
};

// GlobalPoolingResidualBlock - Residual block with global pooling branch
struct GlobalPoolingResidualBlock {
  const string name;
  unique_ptr<NormActConv> preNormActConv;
  unique_ptr<ConvLayer> gpoolConv;
  unique_ptr<BatchNormLayer> gpoolBN;
  unique_ptr<MatMulLayer> gpoolToBiasMul;
  unique_ptr<NormActConv> midNormActConv;
  int numChannels;
  int gpoolChannels;
  int nnXLen;
  int nnYLen;
  bool useFP16;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16_
  ) : name(desc->name),
      numChannels(desc->finalConv.outChannels),
      gpoolChannels(desc->gpoolConv.outChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      useFP16(useFP16_)
  {
    preNormActConv = make_unique<NormActConv>(
      &desc->preBN, &desc->preActivation, &desc->regularConv, nnX, nnY, useFP16_);
    gpoolConv = make_unique<ConvLayer>(&desc->gpoolConv, useFP16_);
    gpoolBN = make_unique<BatchNormLayer>(&desc->gpoolBN, &desc->gpoolActivation, nnX, nnY, useFP16_);
    gpoolToBiasMul = make_unique<MatMulLayer>(&desc->gpoolToBiasMul, useFP16_);
    midNormActConv = make_unique<NormActConv>(
      &desc->midBN, &desc->midActivation, &desc->finalConv, nnX, nnY, useFP16_);
  }

  size_t requiredWorkspaceBytes(int batchSize, aclrtStream stream) const {
    size_t bytes = 0;
    bytes = max(bytes, preNormActConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    bytes = max(bytes, gpoolConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    bytes = max(bytes, midNormActConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    return bytes;
  }

  void apply(
    aclrtStream stream,
    int batchSize,
    const void* maskBuf,
    const float* maskSumBuf,
    void* inputBuf,
    void* regularOutBuf,
    void* gpoolOutBuf,
    void* gpoolConcatBuf,
    void* gpoolBiasBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // This block has two branches:
    // 1. Main branch: input -> preBN -> regularConv -> regularOutBuf
    // 2. Global pooling branch: input -> gpoolConv -> gpoolBN -> global pool -> concat -> matmul -> bias

    // Step 1: Apply pre NormActConv (preBN + preActivation + regularConv)
    // input -> regularOutBuf
    preNormActConv->apply(stream, batchSize, nnXLen, nnYLen, maskBuf, inputBuf, regularOutBuf, workspaceBuf, workspaceBytes);

    // Step 2: Apply gpoolConv on input -> gpoolOutBuf
    gpoolConv->apply(stream, batchSize, nnXLen, nnYLen, false, inputBuf, gpoolOutBuf, workspaceBuf, workspaceBytes);

    // Step 3: Apply gpoolBN on gpoolOutBuf (in-place)
    gpoolBN->apply(stream, batchSize, gpoolOutBuf, maskBuf, gpoolOutBuf, workspaceBuf, workspaceBytes);

    // Step 4: Global pooling (mean, max, mean * sqrt(area))
    // For now, we'll use a simplified global average pooling
    // Full implementation would need:
    // - aclnnAdaptiveAvgPool2d for mean pooling to (1,1)
    // - Custom kernel for max pooling
    // - Concatenation of mean, mean * sqrt(area) - 14, max

    // TODO: Implement proper global pooling with aclnnAdaptiveAvgPool2d
    // For now, use a placeholder that copies zeros
    (void)gpoolConcatBuf;
    (void)maskSumBuf;

    // Step 5: Apply gpoolToBiasMul to get bias for regular branch
    // gpoolConcatBuf (batch, gpoolChannels*3) -> gpoolBiasBuf (batch, numChannels)
    gpoolToBiasMul->apply(stream, batchSize, gpoolConcatBuf, gpoolBiasBuf, workspaceBuf, workspaceBytes);

    // Step 6: Add gpoolBias to regularOutBuf (broadcast bias across spatial dims)
    // TODO: Implement aclnnAdd with proper broadcasting
    // This adds the global pooling bias to each spatial location

    // Step 7: Apply mid NormActConv (midBN + midActivation + finalConv) with residual
    // regularOutBuf -> inputBuf (output, with residual from original input)
    midNormActConv->apply(stream, batchSize, nnXLen, nnYLen, maskBuf, regularOutBuf, inputBuf, workspaceBuf, workspaceBytes);

    // Step 8: Add residual connection
    // TODO: Implement residual add: inputBuf + scratchBuf -> inputBuf
    // Note: The inputBuf should contain the original trunk features for residual

    (void)workspaceBytes;
  }
};

//---------------------------------------------------------------------------------
// Model Structure
//---------------------------------------------------------------------------------

// Forward declarations for head structures
struct Trunk;
struct PolicyHead;
struct ValueHead;

struct Model {
  int numInputChannels;
  int numInputGlobalChannels;
  int numInputMetaChannels;
  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;
  int modelVersion;
  int trunkNumChannels;
  int nnXLen;
  int nnYLen;
  bool useFP16;

  // Trunk layers
  unique_ptr<ConvLayer> initialConv;
  unique_ptr<MatMulLayer> initialMatMul;
  unique_ptr<BatchNormLayer> trunkTipBN;
  vector<unique_ptr<ResidualBlock>> residualBlocks;
  vector<unique_ptr<GlobalPoolingResidualBlock>> gpoolBlocks;

  // Policy head layers
  unique_ptr<ConvLayer> p1Conv;
  unique_ptr<ConvLayer> g1Conv;
  unique_ptr<BatchNormLayer> g1BN;
  unique_ptr<MatMulLayer> gpoolToBiasMul;
  unique_ptr<BatchNormLayer> p1BN;
  unique_ptr<ConvLayer> p2Conv;
  unique_ptr<MatMulLayer> gpoolToPassMul;
  unique_ptr<MatBiasLayer> gpoolToPassBias;
  unique_ptr<MatMulLayer> gpoolToPassMul2;

  // Value head layers
  unique_ptr<ConvLayer> v1Conv;
  unique_ptr<BatchNormLayer> v1BN;
  unique_ptr<MatMulLayer> v2Mul;
  unique_ptr<MatBiasLayer> v2Bias;
  unique_ptr<MatMulLayer> v3Mul;
  unique_ptr<MatBiasLayer> v3Bias;
  unique_ptr<MatMulLayer> sv3Mul;
  unique_ptr<MatBiasLayer> sv3Bias;
  unique_ptr<ConvLayer> vOwnershipConv;

  // SGF Metadata encoder (optional)
  bool hasMetadataEncoder;
  unique_ptr<MatMulLayer> metaMul1;
  unique_ptr<MatBiasLayer> metaBias1;
  unique_ptr<MatMulLayer> metaMul2;
  unique_ptr<MatBiasLayer> metaBias2;
  unique_ptr<MatMulLayer> metaMul3;

  Model(const ModelDesc& desc, int nnX, int nnY, bool fp16);
  ~Model() {}

  void apply(
    aclrtStream stream,
    int batchSize,
    bool requireExactNNLen,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    float* policyPassBuf,
    float* policyBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

  size_t requiredWorkspaceBytes(int maxBatchSize) const;

private:
  void applyTrunk(
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    void* trunkOutputBuf,
    void* maskBuf,
    void* scratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

  void applyPolicyHead(
    aclrtStream stream,
    int batchSize,
    const void* trunkOutputBuf,
    const void* maskBuf,
    float* policyPassBuf,
    float* policyBuf,
    void* scratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

  void applyValueHead(
    aclrtStream stream,
    int batchSize,
    const void* trunkOutputBuf,
    const void* maskBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* scratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;
};

Model::Model(const ModelDesc& desc, int nnX, int nnY, bool fp16)
  : numInputChannels(desc.numInputChannels),
    numInputGlobalChannels(desc.numInputGlobalChannels),
    numInputMetaChannels(desc.numInputMetaChannels),
    numPolicyChannels(desc.numPolicyChannels),
    numValueChannels(desc.numValueChannels),
    numScoreValueChannels(desc.numScoreValueChannels),
    numOwnershipChannels(desc.numOwnershipChannels),
    modelVersion(desc.modelVersion),
    trunkNumChannels(desc.trunk.trunkNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    useFP16(fp16),
    hasMetadataEncoder(desc.numInputMetaChannels > 0)
{
  // Create trunk layers
  initialConv = make_unique<ConvLayer>(&desc.trunk.initialConv, fp16);
  initialMatMul = make_unique<MatMulLayer>(&desc.trunk.initialMatMul, fp16);

  // Create residual blocks
  for(const auto& blockPair : desc.trunk.blocks) {
    int blockKind = blockPair.first;
    if(blockKind == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* rdesc = static_cast<const ResidualBlockDesc*>(blockPair.second.get());
      residualBlocks.push_back(make_unique<ResidualBlock>(rdesc, nnX, nnY, fp16));
    } else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* gdesc = static_cast<const GlobalPoolingResidualBlockDesc*>(blockPair.second.get());
      gpoolBlocks.push_back(make_unique<GlobalPoolingResidualBlock>(gdesc, nnX, nnY, fp16));
    }
  }

  // Create trunk tip BN
  trunkTipBN = make_unique<BatchNormLayer>(&desc.trunk.trunkTipBN, &desc.trunk.trunkTipActivation, nnX, nnY, fp16);

  // Create policy head layers
  p1Conv = make_unique<ConvLayer>(&desc.policyHead.p1Conv, fp16);
  g1Conv = make_unique<ConvLayer>(&desc.policyHead.g1Conv, fp16);
  g1BN = make_unique<BatchNormLayer>(&desc.policyHead.g1BN, &desc.policyHead.g1Activation, nnX, nnY, fp16);
  gpoolToBiasMul = make_unique<MatMulLayer>(&desc.policyHead.gpoolToBiasMul, fp16);
  p1BN = make_unique<BatchNormLayer>(&desc.policyHead.p1BN, &desc.policyHead.p1Activation, nnX, nnY, fp16);
  p2Conv = make_unique<ConvLayer>(&desc.policyHead.p2Conv, fp16);
  gpoolToPassMul = make_unique<MatMulLayer>(&desc.policyHead.gpoolToPassMul, fp16);
  gpoolToPassBias = make_unique<MatBiasLayer>(&desc.policyHead.gpoolToPassBias, fp16);
  gpoolToPassMul2 = make_unique<MatMulLayer>(&desc.policyHead.gpoolToPassMul2, fp16);

  // Create value head layers
  v1Conv = make_unique<ConvLayer>(&desc.valueHead.v1Conv, fp16);
  v1BN = make_unique<BatchNormLayer>(&desc.valueHead.v1BN, &desc.valueHead.v1Activation, nnX, nnY, fp16);
  v2Mul = make_unique<MatMulLayer>(&desc.valueHead.v2Mul, fp16);
  v2Bias = make_unique<MatBiasLayer>(&desc.valueHead.v2Bias, fp16);
  v3Mul = make_unique<MatMulLayer>(&desc.valueHead.v3Mul, fp16);
  v3Bias = make_unique<MatBiasLayer>(&desc.valueHead.v3Bias, fp16);
  sv3Mul = make_unique<MatMulLayer>(&desc.valueHead.sv3Mul, fp16);
  sv3Bias = make_unique<MatBiasLayer>(&desc.valueHead.sv3Bias, fp16);
  vOwnershipConv = make_unique<ConvLayer>(&desc.valueHead.vOwnershipConv, fp16);

  // Create metadata encoder layers if present
  if(hasMetadataEncoder && desc.trunk.sgfMetadataEncoder.metaEncoderVersion > 0) {
    const auto& meta = desc.trunk.sgfMetadataEncoder;
    metaMul1 = make_unique<MatMulLayer>(&meta.mul1, fp16);
    metaBias1 = make_unique<MatBiasLayer>(&meta.bias1, fp16);
    metaMul2 = make_unique<MatMulLayer>(&meta.mul2, fp16);
    metaBias2 = make_unique<MatBiasLayer>(&meta.bias2, fp16);
    metaMul3 = make_unique<MatMulLayer>(&meta.mul3, fp16);
  }
}

size_t Model::requiredWorkspaceBytes(int maxBatchSize) const {
  // Calculate maximum workspace needed across all operations
  size_t maxBytes = 0;

  // Initial conv workspace
  maxBytes = max(maxBytes, initialConv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));

  // Residual blocks
  for(const auto& block : residualBlocks) {
    maxBytes = max(maxBytes, block->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));
  }

  // Policy head
  maxBytes = max(maxBytes, p1Conv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));
  maxBytes = max(maxBytes, p2Conv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));

  // Value head
  maxBytes = max(maxBytes, v1Conv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));
  maxBytes = max(maxBytes, vOwnershipConv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));

  // Add extra for intermediate tensors
  maxBytes += maxBatchSize * trunkNumChannels * nnXLen * nnYLen * sizeof(float) * 4;

  return maxBytes;
}

void Model::apply(
  aclrtStream stream,
  int batchSize,
  bool requireExactNNLen,
  void* inputBuf,
  void* inputGlobalBuf,
  void* inputMetaBuf,
  float* policyPassBuf,
  float* policyBuf,
  float* valueBuf,
  float* scoreValueBuf,
  void* ownershipBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {
  // Allocate intermediate buffers
  size_t eltSize = useFP16 ? sizeof(aclFloat16) : sizeof(float);
  size_t trunkOutputBytes = (size_t)batchSize * trunkNumChannels * nnXLen * nnYLen * eltSize;
  size_t maskBytes = (size_t)batchSize * 1 * nnXLen * nnYLen * eltSize;
  size_t scratchBytes = trunkOutputBytes;

  void* trunkOutputBuf = ascendMalloc(trunkOutputBytes);
  void* maskBuf = ascendMalloc(maskBytes);
  void* scratchBuf = ascendMalloc(scratchBytes);

  // Apply trunk
  applyTrunk(stream, batchSize, inputBuf, inputGlobalBuf, inputMetaBuf, trunkOutputBuf, maskBuf, scratchBuf, workspaceBuf, workspaceBytes);

  // Apply policy head
  applyPolicyHead(stream, batchSize, trunkOutputBuf, maskBuf, policyPassBuf, policyBuf, scratchBuf, workspaceBuf, workspaceBytes);

  // Apply value head
  applyValueHead(stream, batchSize, trunkOutputBuf, maskBuf, valueBuf, scoreValueBuf, ownershipBuf, scratchBuf, workspaceBuf, workspaceBytes);

  // Free intermediate buffers
  ascendFree(scratchBuf);
  ascendFree(maskBuf);
  ascendFree(trunkOutputBuf);

  (void)requireExactNNLen;
}

void Model::applyTrunk(
  aclrtStream stream,
  int batchSize,
  void* inputBuf,
  void* inputGlobalBuf,
  void* inputMetaBuf,
  void* trunkOutputBuf,
  void* maskBuf,
  void* scratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {
  // Apply initial convolution: input -> trunkOutput
  initialConv->apply(stream, batchSize, nnXLen, nnYLen, false, inputBuf, trunkOutputBuf, workspaceBuf, workspaceBytes);

  // Apply initial matmul with global features
  // This adds the global features (and optional metadata features) to trunkOutput
  // The global features are broadcast and added to each spatial location
  // KataGo: trunkOutput += inputGlobalBuf @ initialMatMul

  // Allocate temporary buffer for matmul result
  void* globalMulBuf = scratchBuf;  // Reuse scratch buffer for intermediate result

  // Apply matmul: (batch, numGlobalChannels) @ (numGlobalChannels, trunkChannels) -> (batch, trunkChannels)
  initialMatMul->apply(stream, batchSize, inputGlobalBuf, globalMulBuf, workspaceBuf, workspaceBytes);

  // Add globalMulBuf to each spatial location of trunkOutput
  // trunkOutput is (batch, trunkChannels, nnYLen, nnXLen)
  // globalMulBuf is (batch, trunkChannels)
  // Need to broadcast add: trunkOutput += globalMulBuf[:, :, None, None]

  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

  // Create tensors for add with broadcasting
  vector<int64_t> trunkShape = {batchSize, trunkNumChannels, nnYLen, nnXLen};
  vector<int64_t> biasShape = {batchSize, trunkNumChannels, 1, 1};  // Broadcast to NCHW

  aclTensor* trunkTensor = createAclTensor(trunkOutputBuf, trunkShape, dtype, ACL_FORMAT_NCHW);
  aclTensor* biasTensor = createAclTensor(globalMulBuf, biasShape, dtype, ACL_FORMAT_NCHW);
  aclTensor* resultTensor = createAclTensor(trunkOutputBuf, trunkShape, dtype, ACL_FORMAT_NCHW);

  // Create scalar for alpha
  aclScalar* alpha = createFloatScalar(1.0f);

  // Phase 1: Get workspace size for add
  uint64_t addWsSize = 0;
  aclOpExecutor* addExecutor = nullptr;

  aclnnStatus status = aclnnAddGetWorkspaceSize(trunkTensor, biasTensor, alpha, resultTensor, &addWsSize, &addExecutor);

  if(status == ACLNN_SUCCESS) {
    // Phase 2: Execute add
    status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
  }

  // Cleanup
  destroyAclTensor(trunkTensor);
  destroyAclTensor(biasTensor);
  destroyAclTensor(resultTensor);
  aclDestroyScalar(alpha);

  if(status != ACLNN_SUCCESS) {
    throw StringError("aclnnAdd failed for initial global features with error: " + to_string(status));
  }

  // TODO: Handle metadata features if present
  (void)inputMetaBuf;
  (void)hasMetadataEncoder;
  (void)metaMul1;
  (void)metaBias1;
  (void)metaMul2;
  (void)metaBias2;
  (void)metaMul3;

  // Extract mask from input channel 0
  // The mask is stored in the first channel of the input
  // TODO: Implement mask extraction using aclnn indexing or stride
  (void)maskBuf;

  // Apply residual blocks
  for(const auto& block : residualBlocks) {
    block->apply(stream, batchSize, nnXLen, nnYLen, maskBuf, trunkOutputBuf, trunkOutputBuf, scratchBuf, workspaceBuf, workspaceBytes);
  }

  // Apply global pooling residual blocks
  // These require mask sum buffer for proper global pooling
  // TODO: Implement gpoolBlocks with maskSumBuf

  // Apply trunk tip BN
  trunkTipBN->apply(stream, batchSize, trunkOutputBuf, maskBuf, trunkOutputBuf, workspaceBuf, workspaceBytes);

  (void)scratchBuf;
}

void Model::applyPolicyHead(
  aclrtStream stream,
  int batchSize,
  const void* trunkOutputBuf,
  const void* maskBuf,
  float* policyPassBuf,
  float* policyBuf,
  void* scratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {
  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  int p1Channels = p1Conv->outChannels;
  int g1Channels = g1Conv->outChannels;
  size_t eltSize = useFP16 ? sizeof(aclFloat16) : sizeof(float);

  // Allocate intermediate buffers
  size_t p1OutBytes = (size_t)batchSize * p1Channels * nnXLen * nnYLen * eltSize;
  size_t g1OutBytes = (size_t)batchSize * g1Channels * nnXLen * nnYLen * eltSize;
  size_t g1ConcatBytes = (size_t)batchSize * g1Channels * 3 * sizeof(float);
  size_t g1BiasBytes = (size_t)batchSize * p1Channels * sizeof(float);
  size_t p1PassBytes = (size_t)batchSize * p1Channels * sizeof(float);

  void* p1OutBuf = ascendMalloc(p1OutBytes);
  void* g1OutBuf = ascendMalloc(g1OutBytes);
  void* g1Out2Buf = ascendMalloc(g1OutBytes);
  void* g1ConcatBuf = ascendMalloc(g1ConcatBytes);
  void* g1BiasBuf = ascendMalloc(g1BiasBytes);
  void* p1PassBuf = ascendMalloc(p1PassBytes);

  // Step 1: Apply p1Conv: trunk -> p1Out
  p1Conv->apply(stream, batchSize, nnXLen, nnYLen, false, const_cast<void*>(trunkOutputBuf), p1OutBuf, workspaceBuf, workspaceBytes);

  // Step 2: Apply g1Conv: trunk -> g1Out
  g1Conv->apply(stream, batchSize, nnXLen, nnYLen, false, const_cast<void*>(trunkOutputBuf), g1OutBuf, workspaceBuf, workspaceBytes);

  // Step 3: Apply g1BN: g1Out -> g1Out2
  g1BN->apply(stream, batchSize, g1OutBuf, maskBuf, g1Out2Buf, workspaceBuf, workspaceBytes);

  // Step 4: Global pooling on g1Out2 -> g1Concat
  // This computes: mean, max, mean * sqrt(area) and concatenates them
  // TODO: Implement proper global pooling using aclnnAdaptiveAvgPool2d + custom max kernel
  // For now, zero-initialize g1ConcatBuf as a placeholder
  // In practice, we need custom kernels for this or implement with ACLNN operations

  // Step 5: Apply gpoolToBiasMul: g1Concat -> g1Bias
  gpoolToBiasMul->apply(stream, batchSize, g1ConcatBuf, g1BiasBuf, workspaceBuf, workspaceBytes);

  // Step 6: Add g1Bias to p1Out (broadcast across spatial dims)
  // g1Bias is (batch, p1Channels, 1, 1), p1Out is (batch, p1Channels, nnYLen, nnXLen)
  {
    vector<int64_t> p1OutShape = {batchSize, p1Channels, nnYLen, nnXLen};
    vector<int64_t> biasShape = {batchSize, p1Channels, 1, 1};  // Broadcast to NCHW

    aclTensor* p1OutTensor = createAclTensor(p1OutBuf, p1OutShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* biasTensor = createAclTensor(g1BiasBuf, biasShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* resultTensor = createAclTensor(p1OutBuf, p1OutShape, dtype, ACL_FORMAT_NCHW);
    aclScalar* alpha = createFloatScalar(1.0f);

    uint64_t addWsSize = 0;
    aclOpExecutor* addExecutor = nullptr;
    aclnnStatus status = aclnnAddGetWorkspaceSize(p1OutTensor, biasTensor, alpha, resultTensor, &addWsSize, &addExecutor);
    if(status == ACLNN_SUCCESS && addWsSize <= workspaceBytes) {
      aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
    }

    destroyAclTensor(p1OutTensor);
    destroyAclTensor(biasTensor);
    destroyAclTensor(resultTensor);
    aclDestroyScalar(alpha);

    if(status != ACLNN_SUCCESS) {
      ascendFree(p1OutBuf); ascendFree(g1OutBuf); ascendFree(g1Out2Buf);
      ascendFree(g1ConcatBuf); ascendFree(g1BiasBuf); ascendFree(p1PassBuf);
      throw StringError("aclnnAdd failed for policy head bias with error: " + to_string(status));
    }
  }

  // Step 7: Apply p1BN: p1Out -> scratchBuf
  p1BN->apply(stream, batchSize, p1OutBuf, maskBuf, scratchBuf, workspaceBuf, workspaceBytes);

  // Step 8: Apply p2Conv: scratchBuf -> policyBuf
  p2Conv->apply(stream, batchSize, nnXLen, nnYLen, false, scratchBuf, policyBuf, workspaceBuf, workspaceBytes);

  // Step 9: Compute policy pass logit
  // gpoolToPassMul: g1Concat -> p1PassBuf
  // gpoolToPassBias: p1PassBuf (in-place)
  // gpoolToPassMul2: p1PassBuf -> policyPassBuf (if modelVersion >= 15)
  if(modelVersion >= 15) {
    gpoolToPassMul->apply(stream, batchSize, g1ConcatBuf, p1PassBuf, workspaceBuf, workspaceBytes);
    gpoolToPassBias->apply(stream, batchSize, p1PassBuf, p1PassBuf, workspaceBuf, workspaceBytes);
    gpoolToPassMul2->apply(stream, batchSize, p1PassBuf, policyPassBuf, workspaceBuf, workspaceBytes);
  } else {
    gpoolToPassMul->apply(stream, batchSize, g1ConcatBuf, policyPassBuf, workspaceBuf, workspaceBytes);
  }

  // Free intermediate buffers
  ascendFree(p1PassBuf);
  ascendFree(g1BiasBuf);
  ascendFree(g1ConcatBuf);
  ascendFree(g1Out2Buf);
  ascendFree(g1OutBuf);
  ascendFree(p1OutBuf);

  (void)dtype;
}

void Model::applyValueHead(
  aclrtStream stream,
  int batchSize,
  const void* trunkOutputBuf,
  const void* maskBuf,
  float* valueBuf,
  float* scoreValueBuf,
  void* ownershipBuf,
  void* scratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {
  (void)scratchBuf;  // Not currently used, but kept for API compatibility
  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  int v1Channels = v1Conv->outChannels;
  int v2Channels = v2Mul->outChannels;
  int ownershipChannels = vOwnershipConv->outChannels;
  size_t eltSize = useFP16 ? sizeof(aclFloat16) : sizeof(float);

  // Allocate intermediate buffers
  size_t v1OutBytes = (size_t)batchSize * v1Channels * nnXLen * nnYLen * eltSize;
  size_t v1MeanBytes = (size_t)batchSize * v1Channels * 3 * sizeof(float);  // mean, max, sqrt-area
  size_t v2OutBytes = (size_t)batchSize * v2Channels * sizeof(float);
  size_t ownershipScratchBytes = (size_t)batchSize * ownershipChannels * nnXLen * nnYLen * eltSize;

  void* v1OutBuf = ascendMalloc(v1OutBytes);
  void* v1Out2Buf = ascendMalloc(v1OutBytes);
  void* v1MeanBuf = ascendMalloc(v1MeanBytes);
  void* v2OutBuf = ascendMalloc(v2OutBytes);
  void* ownershipScratchBuf = ascendMalloc(ownershipScratchBytes);

  // Step 1: Apply v1Conv: trunk -> v1Out
  v1Conv->apply(stream, batchSize, nnXLen, nnYLen, false, const_cast<void*>(trunkOutputBuf), v1OutBuf, workspaceBuf, workspaceBytes);

  // Step 2: Apply v1BN: v1Out -> v1Out2
  v1BN->apply(stream, batchSize, v1OutBuf, maskBuf, v1Out2Buf, workspaceBuf, workspaceBytes);

  // Step 3: Global pooling on v1Out2 -> v1Mean
  // TODO: Implement proper global pooling with mean, max, sqrt-area statistics
  // For now, use aclnnAdaptiveAvgPool2d for mean only
  // The CUDA backend uses customCudaValueHeadPoolNCHW which computes:
  //   mean, max, and mean * sqrt(area) concatenated
  // We need to implement this properly or create a custom kernel

  // For now, zero-initialize v1MeanBuf using aclnnFillScalar
  {
    aclTensor* v1MeanTensor = createAclTensor(v1MeanBuf, {batchSize, v1Channels * 3}, ACL_FLOAT, ACL_FORMAT_ND);
    aclScalar* zeroScalar = createAclScalar(0.0f, ACL_FLOAT);

    uint64_t fillWsSize = 0;
    aclOpExecutor* fillExecutor = nullptr;
    aclnnStatus status = aclnnInplaceFillScalarGetWorkspaceSize(v1MeanTensor, zeroScalar, &fillWsSize, &fillExecutor);
    if(status == ACLNN_SUCCESS) {
      status = aclnnInplaceFillScalar(workspaceBuf, fillWsSize, fillExecutor, stream);
    }

    destroyAclTensor(v1MeanTensor);
    aclDestroyScalar(zeroScalar);

    if(status != ACLNN_SUCCESS) {
      ascendFree(v1OutBuf); ascendFree(v1Out2Buf); ascendFree(v1MeanBuf);
      ascendFree(v2OutBuf); ascendFree(ownershipScratchBuf);
      throw StringError("aclnnInplaceFillScalar failed for v1MeanBuf with error: " + to_string(status));
    }
  }

  // Step 4: Apply v2Mul: v1Mean -> v2Out
  v2Mul->apply(stream, batchSize, v1MeanBuf, v2OutBuf, workspaceBuf, workspaceBytes);

  // Step 5: Apply v2Bias: v2Out -> v2Out (in-place)
  v2Bias->apply(stream, batchSize, v2OutBuf, v2OutBuf, workspaceBuf, workspaceBytes);

  // Step 6: Apply v3Mul: v2Out -> valueBuf
  v3Mul->apply(stream, batchSize, v2OutBuf, valueBuf, workspaceBuf, workspaceBytes);

  // Step 7: Apply v3Bias: valueBuf -> valueBuf (in-place)
  v3Bias->apply(stream, batchSize, valueBuf, valueBuf, workspaceBuf, workspaceBytes);

  // Step 8: Apply sv3Mul: v2Out -> scoreValueBuf
  sv3Mul->apply(stream, batchSize, v2OutBuf, scoreValueBuf, workspaceBuf, workspaceBytes);

  // Step 9: Apply sv3Bias: scoreValueBuf -> scoreValueBuf (in-place)
  sv3Bias->apply(stream, batchSize, scoreValueBuf, scoreValueBuf, workspaceBuf, workspaceBytes);

  // Step 10: Apply vOwnershipConv: v1Out2 -> ownershipBuf
  // If using FP16, output to ownershipScratchBuf first, then convert to FP32
  if(useFP16) {
    vOwnershipConv->apply(stream, batchSize, nnXLen, nnYLen, false, v1Out2Buf, ownershipScratchBuf, workspaceBuf, workspaceBytes);
    // Convert FP16 to FP32 for final output
    aclTensor* srcTensor = createAclTensor(ownershipScratchBuf, {batchSize, ownershipChannels, nnYLen, nnXLen}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    aclTensor* dstTensor = createAclTensor(ownershipBuf, {batchSize, ownershipChannels, nnYLen, nnXLen}, ACL_FLOAT, ACL_FORMAT_NCHW);

    uint64_t castWsSize = 0;
    aclOpExecutor* castExecutor = nullptr;
    aclnnStatus status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT, dstTensor, &castWsSize, &castExecutor);
    if(status == ACLNN_SUCCESS && castWsSize <= workspaceBytes) {
      aclnnCast(workspaceBuf, castWsSize, castExecutor, stream);
    }
    destroyAclTensor(srcTensor);
    destroyAclTensor(dstTensor);
  } else {
    vOwnershipConv->apply(stream, batchSize, nnXLen, nnYLen, false, v1Out2Buf, ownershipBuf, workspaceBuf, workspaceBytes);
  }

  // Free intermediate buffers
  ascendFree(ownershipScratchBuf);
  ascendFree(v2OutBuf);
  ascendFree(v1MeanBuf);
  ascendFree(v1Out2Buf);
  ascendFree(v1OutBuf);

  (void)dtype;
}

//---------------------------------------------------------------------------------
// ComputeContext / ComputeHandle creation
//---------------------------------------------------------------------------------

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  vector<int> actualGpuIdxs = gpuIdxs;
  if(actualGpuIdxs.size() <= 0 || (actualGpuIdxs.size() == 1 && actualGpuIdxs[0] == -1)) {
    actualGpuIdxs = {0};
  }

  // Set default device before any allocations (model weights, buffers, etc.)
  int defaultDevice = actualGpuIdxs[0];
  aclError ret = aclrtSetDevice(defaultDevice);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtSetDevice failed in createComputeContext for device " + to_string(defaultDevice) + " with error: " + to_string(ret));
  }

  return new ComputeContext(nnXLen, nnYLen, useFP16Mode, useNHWCMode, actualGpuIdxs);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)serverThreadIdx;

  int deviceIdx = (gpuIdxForThisThread == -1) ? 0 : gpuIdxForThisThread;

  // Determine FP16 mode - Ascend 910ProA is optimized for FP16
  // Enable by default for maximum performance
  bool useFP16 = true;
  if(context->useFP16Mode == enabled_t::True)
    useFP16 = true;
  else if(context->useFP16Mode == enabled_t::False)
    useFP16 = false;
  // else auto -> use FP16 (already set to true above)

  ComputeHandle* handle = new ComputeHandle(
    deviceIdx, useFP16, context->nnXLen, context->nnYLen, requireExactNNLen, inputsUseNHWC
  );

  // Set device and create stream
  aclError ret = aclrtSetDevice(deviceIdx);
  if(ret != ACL_SUCCESS) {
    delete handle;
    throw StringError("aclrtSetDevice failed for device " + to_string(deviceIdx) + " with error: " + to_string(ret));
  }

  ret = aclrtCreateStream(&handle->stream);
  if(ret != ACL_SUCCESS) {
    delete handle;
    throw StringError("aclrtCreateStream failed with error: " + to_string(ret));
  }

  // Log device assignment for multi-NPU debugging
  if(logger != nullptr) {
    logger->write(
      "Ascend NPU backend thread " + Global::intToString(serverThreadIdx)
      + ": using device " + Global::intToString(deviceIdx)
      + ", FP16 " + string(useFP16 ? "enabled" : "disabled")
    );
  }

  // Create model
  const ModelDesc& modelDesc = loadedModel->modelDesc;
  Model* model = new Model(modelDesc, context->nnXLen, context->nnYLen, useFP16);
  handle->model = model;

  // Create scratch buffers
  size_t workspaceNeeded = model->requiredWorkspaceBytes(maxBatchSize);
  // Ensure minimum workspace for multi-NPU stability - 256MB minimum
  // This prevents resource exhaustion when many ops are queued simultaneously
  size_t minWorkspace = 256 * 1024 * 1024;
  workspaceNeeded = std::max(workspaceNeeded, minWorkspace);
  ScratchBuffers* scratch = new ScratchBuffers(maxBatchSize, context->nnXLen, context->nnYLen, useFP16, workspaceNeeded);
  handle->scratch = scratch;

  // Create device buffers
  Buffers* buffers = new Buffers(modelDesc, maxBatchSize, context->nnXLen, context->nnYLen, useFP16, workspaceNeeded);
  handle->buffers = buffers;

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  if(computeHandle != nullptr) {
    // Set device context before freeing resources on the correct device
    aclrtSetDevice(computeHandle->deviceIdx);
    delete computeHandle->buffers;
    delete computeHandle->scratch;
    delete computeHandle->model;
    delete computeHandle;
  }
}

bool NeuralNet::isUsingFP16(const ComputeHandle* computeHandle) {
  return computeHandle->usingFP16;
}

//---------------------------------------------------------------------------------
// getOutput
//---------------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  // Set device context - critical for multi-NPU since each server thread
  // needs to be bound to its device for all ACL/ACLNN calls
  aclrtSetDevice(gpuHandle->deviceIdx);

  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->model->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = (int)inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = gpuHandle->model->numPolicyChannels;

  // Copy inputs from individual NNResultBufs to batched host buffers
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->userInputMetaBuffer + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;

    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);

    if(numMetaFeatures > 0) {
      testAssert(rowMeta != nullptr);
      testAssert(hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    } else {
      testAssert(!hasRowMeta);
    }

    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures,
      gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry
    );
  }

  Buffers* buffers = gpuHandle->buffers;

  // Copy host buffers to device
  if(!gpuHandle->usingFP16) {
    ascendCopyH2D(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes * batchSize);
    ascendCopyH2D(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes * batchSize);
    if(numMetaFeatures > 0) {
      ascendCopyH2D(buffers->inputMetaBuf, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes * batchSize);
    }
  } else {
    // For FP16 mode, copy to float buffer first, then convert to FP16
    ascendCopyH2D(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes * batchSize);
    ascendCopyH2D(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes * batchSize);
    if(numMetaFeatures > 0) {
      ascendCopyH2D(buffers->inputMetaBufFloat, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes * batchSize);
    }

    // Convert float to FP16 on device using aclnnCast

    // Convert spatial input
    {
      aclTensor* srcTensor = createAclTensor(buffers->inputBufFloat,
        {batchSize, numSpatialFeatures, nnYLen, nnXLen}, ACL_FLOAT, ACL_FORMAT_NCHW);
      aclTensor* dstTensor = createAclTensor(buffers->inputBuf,
        {batchSize, numSpatialFeatures, nnYLen, nnXLen}, ACL_FLOAT16, ACL_FORMAT_NCHW);

      uint64_t castWsSize = 0;
      aclOpExecutor* castExecutor = nullptr;
      aclnnStatus status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT16, dstTensor, &castWsSize, &castExecutor);
      if(status == ACLNN_SUCCESS) {
        status = aclnnCast(buffers->workspaceBuf, castWsSize, castExecutor, gpuHandle->stream);
      }
      destroyAclTensor(srcTensor);
      destroyAclTensor(dstTensor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnCast failed for spatial input with error: " + to_string(status));
      }
    }

    // Convert global input
    {
      aclTensor* srcTensor = createAclTensor(buffers->inputGlobalBufFloat,
        {batchSize, numGlobalFeatures}, ACL_FLOAT, ACL_FORMAT_ND);
      aclTensor* dstTensor = createAclTensor(buffers->inputGlobalBuf,
        {batchSize, numGlobalFeatures}, ACL_FLOAT16, ACL_FORMAT_ND);

      uint64_t castWsSize = 0;
      aclOpExecutor* castExecutor = nullptr;
      aclnnStatus status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT16, dstTensor, &castWsSize, &castExecutor);
      if(status == ACLNN_SUCCESS) {
        status = aclnnCast(buffers->workspaceBuf, castWsSize, castExecutor, gpuHandle->stream);
      }
      destroyAclTensor(srcTensor);
      destroyAclTensor(dstTensor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnCast failed for global input with error: " + to_string(status));
      }
    }

    // Convert meta input if present
    if(numMetaFeatures > 0) {
      aclTensor* srcTensor = createAclTensor(buffers->inputMetaBufFloat,
        {batchSize, numMetaFeatures}, ACL_FLOAT, ACL_FORMAT_ND);
      aclTensor* dstTensor = createAclTensor(buffers->inputMetaBuf,
        {batchSize, numMetaFeatures}, ACL_FLOAT16, ACL_FORMAT_ND);

      uint64_t castWsSize = 0;
      aclOpExecutor* castExecutor = nullptr;
      aclnnStatus status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT16, dstTensor, &castWsSize, &castExecutor);
      if(status == ACLNN_SUCCESS) {
        status = aclnnCast(buffers->workspaceBuf, castWsSize, castExecutor, gpuHandle->stream);
      }
      destroyAclTensor(srcTensor);
      destroyAclTensor(dstTensor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnCast failed for meta input with error: " + to_string(status));
      }
    }
  }

  // Run model inference
  gpuHandle->model->apply(
    gpuHandle->stream,
    batchSize,
    gpuHandle->requireExactNNLen,
    buffers->inputBuf,
    buffers->inputGlobalBuf,
    buffers->inputMetaBuf,
    buffers->policyPassBuf,
    buffers->policyBuf,
    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,
    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  // Synchronize before copying results back
  // This is critical for multi-NPU stability - ensures all ops complete before D2H copies
  aclError ret = aclrtSynchronizeStream(gpuHandle->stream);
  if(ret != ACL_SUCCESS) {
    // Error 507015 typically indicates stream timeout or resource exhaustion
    // Log detailed error for debugging multi-NPU issues
    string errMsg = "aclrtSynchronizeStream failed on device " + to_string(gpuHandle->deviceIdx)
      + " with error: " + to_string((int)ret)
      + " (batchSize=" + to_string(batchSize) + ")";
    throw StringError(errMsg);
  }

  // Copy results back to host
  ascendCopyD2H(inputBuffers->policyPassResults, buffers->policyPassBuf, inputBuffers->singlePolicyPassResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes * batchSize);

  // Extract results into NNOutput structs
  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = inputBuffers->policyPassResults + row * numPolicyChannels;
    const float* policySrcBuf = inputBuffers->policyResults + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // Handle policy with optimism
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      if(gpuHandle->inputsUseNHWC) {
        for(int i = 0; i < nnXLen * nnYLen; i++) {
          float p = policySrcBuf[i * numPolicyChannels];
          float pOpt = policySrcBuf[i * numPolicyChannels + 1];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      } else {
        for(int i = 0; i < nnXLen * nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i + nnXLen * nnYLen];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
    } else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
    }

    // Value outputs
    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    // Ownership
    if(output->whiteOwnerMap != nullptr) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    // Score/value outputs based on model version
    if(modelVersion >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    } else if(modelVersion >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

//---------------------------------------------------------------------------------
// Test functions
//---------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  (void)useNHWC; // We always use NCHW internally

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;

  if(numInputFloats != inputBuffer.size()) {
    throw StringError("testEvaluateConv: unexpected input buffer size");
  }

  // Set device
  ACL_CHECK(aclrtSetDevice(0), "aclrtSetDevice");

  // Create stream
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

  // Create layer
  ConvLayer* convLayer = new ConvLayer(desc, useFP16);

  // Allocate device buffers
  void* deviceInput = ascendMalloc(numInputFloats * (useFP16 ? sizeof(aclFloat16) : sizeof(float)));
  void* deviceOutput = ascendMalloc(numOutputFloats * (useFP16 ? sizeof(aclFloat16) : sizeof(float)));

  // Copy input to device
  ascendCopyH2D(deviceInput, inputBuffer.data(), numInputFloats * sizeof(float));

  // Get workspace size
  size_t workspaceBytes = convLayer->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream);
  void* deviceWorkspace = nullptr;
  if(workspaceBytes > 0) {
    deviceWorkspace = ascendMalloc(workspaceBytes);
  }

  // Apply convolution
  convLayer->apply(stream, batchSize, nnXLen, nnYLen, false, deviceInput, deviceOutput, deviceWorkspace, workspaceBytes);

  // Synchronize
  ACL_CHECK(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

  // Copy output back to host
  outputBuffer.resize(numOutputFloats);
  ascendCopyD2H(outputBuffer.data(), deviceOutput, numOutputFloats * sizeof(float));

  // Cleanup
  ascendFree(deviceWorkspace);
  ascendFree(deviceInput);
  ascendFree(deviceOutput);
  delete convLayer;
  ACL_CHECK(aclrtDestroyStream(stream), "aclrtDestroyStream");

  return true;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;

  return false; // Not implemented yet
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;

  return false; // Not implemented yet
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;

  return false; // Not implemented yet
}

#endif // USE_ASCEND_BACKEND
