# CLAUDE.md - AscendGo (KataGo Ascend NPU Backend)

> **Version: 1.0.0** | Last Updated: 2025-03-10
> **Target Model: GLM5** | Designed for autonomous, agentic operation

---

## Project Overview

AscendGo is a fork of KataGo with a fifth inference backend ("ASCEND") for Huawei Ascend 910 Pro A NPUs. This is **inference-only** - no training/backward pass support needed.

### Key Facts
- **Hardware**: 4x Ascend 910 Pro A (32GB HBM2 each, 256 TFLOPS FP16)
- **Software**: CANN 8.3.RC1.alpha003, aarch64-linux, C++14+
- **Critical Constraint**: No bf16 support - use fp16 or fp32 only
- **Performance Sweet Spot**: FP16 with FP32 accumulation

### Repository Structure
```
cpp/
├── neuralnet/
│   ├── ascendbackend.cpp    # MAIN FILE - Ascend NPU backend implementation
│   ├── cudabackend.cpp      # Reference implementation
│   ├── cudahelpers.cu       # CUDA kernels (reference for ACLNN ops)
│   ├── nninterface.h        # Neural net interface
│   └── desc.h               # Model description structures
├── core/                    # Low-level utilities
├── game/                    # Board representation and rules
├── search/                  # MCTS implementation
└── command/                 # CLI commands (gtp, benchmark, etc.)
```

---

## Build Commands

```bash
# On Ascend hardware (aarch64-linux)
cd cpp && mkdir -p build && cd build
cmake -DUSE_BACKEND=ASCEND -DASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest ..
make -j$(nproc)

# Run basic test
./katago version
./katago benchmark -model <model_file> -config <config_file>
```

---

## Autonomous Error-Fixing Workflow

### Phase 1: Error Detection
When encountering compile or runtime errors:

1. **Parse the error message carefully**
   - ACL error codes: 161002 = dtype mismatch, 107001 = invalid shape, etc.
   - Look for line numbers and function names

2. **Identify the root cause**
   - Dtype mismatch (FP16 vs FP32)
   - Shape mismatch (NCHW vs ND format)
   - Missing include or wrong header
   - Null pointer or uninitialized buffer

3. **Cross-reference with working code**
   - Compare with CUDA backend (`cudabackend.cpp`)
   - Check CUDA helpers (`cudahelpers.cu`) for algorithm reference

### Phase 2: Fix Implementation
1. **Make minimal, targeted changes**
   - Fix only the specific issue
   - Don't refactor unrelated code

2. **Follow existing patterns**
   - Use the same code style
   - Match buffer allocation/deallocation patterns
   - Use tensorCache for aclTensor creation

3. **Add comments for non-obvious fixes**
   - Explain WHY the fix is needed
   - Reference CANN documentation if applicable

### Phase 3: Verification
1. **Commit and push after every fix**
   ```bash
   git add -A && git commit -m "Fix: <description>"
   git push origin master
   ```

2. **Wait for user feedback**
   - User will test on remote Ascend hardware
   - Do NOT build or test locally (no Ascend hardware)

---

## Critical Implementation Patterns

### 1. ACLNN Two-Phase Call Pattern
Every ACLNN operator uses this pattern:
```cpp
// Phase 1: Get workspace size
uint64_t wsSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus status = aclnnXxxGetWorkspaceSize(..., &wsSize, &executor);

// Phase 2: Execute (use preallocated workspace)
if(status == ACLNN_SUCCESS && wsSize <= workspaceBytes) {
  status = aclnnXxx(workspaceBuf, wsSize, executor, stream);
}
```

### 2. Tensor Creation Pattern
Use tensorCache to avoid repeated tensor creation:
```cpp
aclTensor* tensor = handle->tensorCache.get(bufferPtr, {batch, channels, h, w}, dtype, ACL_FORMAT_NCHW);
```

### 3. FP16/FP32 Mixed Precision Pattern
Global pooling always outputs FP32, but FP16 mode requires FP16 intermediate:
```cpp
// Pool to dtype buffer
void* poolBuf = scratchBuf;  // FP16 or FP32 depending on mode
aclTensor* poolTensor = handle->tensorCache.get(poolBuf, shape, dtype, format);
aclnnAdaptiveAvgPool2d(..., poolTensor, ...);

// Cast to FP32 if needed for concatenation
if(useFP16) {
  aclTensor* fp32Tensor = handle->tensorCache.get(fp32Buf, shape, ACL_FLOAT, format);
  aclnnCast(poolTensor, ACL_FLOAT, fp32Tensor, ...);
}
```

### 4. MatMulLayer Input Dtype Pattern
MatMulLayer expects FP16 input when `useFP16=true`:
```cpp
// Global pooling outputs FP32 - MUST cast to FP16 before MatMulLayer
void* inputBuf;
if(useFP16) {
  aclnnCast(fp32Buf, ACL_FLOAT16, fp16Buf, ...);
  inputBuf = fp16Buf;
} else {
  inputBuf = fp32Buf;
}
matMulLayer->apply(..., inputBuf, outputBuf, ...);
```

---

## Known Issues & Solutions

### Issue: ACL Error 161002 (Dtype Mismatch)
**Cause**: Input and output tensors have different dtypes for operators that require matching dtypes.

**Solution**: Ensure all pooling operations use the same dtype for input and output:
```cpp
// WRONG: Input is dtype, output is ACL_FLOAT
aclTensor* outTensor = handle->tensorCache.get(buf, shape, ACL_FLOAT, format);

// CORRECT: Both use dtype
aclTensor* outTensor = handle->tensorCache.get(buf, shape, dtype, format);
```

### Issue: aclnnMuls Include Path
**Cause**: `aclnn_muls.h` doesn't exist; `aclnnMuls` is in `aclnn_mul.h`.

**Solution**:
```cpp
#include "aclnnop/aclnn_mul.h"  // NOT "aclnn_muls.h"
```

### Issue: aclCreateIntArray Brace Initialization
**Cause**: g++ 7.3 doesn't support brace-enclosed initializer list for C-style functions.

**Solution**: Use helper function:
```cpp
aclIntArray* arr = createAclIntArray({1, 2, 3});  // NOT aclCreateIntArray({1, 2, 3})
```

### Issue: FP32 Data Interpreted as FP16
**Cause**: FP32 buffer passed to layer that creates FP16 tensor.

**Solution**: Add explicit FP32->FP16 cast before the layer:
```cpp
if(useFP16) {
  aclnnCast(fp32SrcTensor, ACL_FLOAT16, fp16DstTensor, ...);
  inputForLayer = fp16DstTensor;
}
```

---

## Buffer Layout Reference

### Policy Head Global Pooling (FP16 mode)
```
Scratch buffer layout:
[meanFP16][meanFP32][maxFP16][maxFP32][scaledMeanFP32]
```

### Value Head Global Pooling (FP16 mode)
```
Scratch buffer layout:
[meanFP16][meanFP32][scaledMean1FP32][scaledMean2FP32]
```

---

## Git Workflow

### Commit Message Format
```
<type>: <short description>

<optional longer description>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Types: `Fix`, `Implement`, `Refactor`, `Add`, `Update`

### After Every Change
```bash
git add -A
git commit -m "..."
git push origin master
```

---

## Reference Files

| File | Purpose |
|------|---------|
| `cudabackend.cpp` | Reference implementation for all backends |
| `cudahelpers.cu` | CUDA kernels - algorithm reference for ACLNN ops |
| `nninterface.h` | Interface that must be implemented |
| `desc.h` | Model structure definitions |

---

## Autonomous Analysis Checklist

Before declaring code complete, verify:

- [ ] All ACLNN calls use two-phase pattern (GetWorkspaceSize + Execute)
- [ ] All tensor dtypes match where required (pooling ops)
- [ ] FP16 mode has proper FP32->FP16 casts before MatMulLayer
- [ ] Buffer allocations match deallocations in destructor
- [ ] No brace-enclosed initializers for C functions
- [ ] All includes use correct paths (`aclnnop/` prefix)
- [ ] Tensor shapes are correct (NCHW for spatial, ND for 2D)
- [ ] Workspace size checks include `<= workspaceBytes` condition

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-03-10 | Initial version for GLM5 |

---

## Instructions for AI Assistants

1. **DO NOT build or test** - User tests on remote Ascend hardware
2. **ALWAYS push to GitHub** after every meaningful change
3. **Follow existing code patterns** - Look at similar code in the same file
4. **Cross-reference CUDA backend** for algorithm correctness
5. **Add comments** for non-obvious dtype/shape decisions
6. **Keep commits atomic** - One fix per commit when possible

---

*This file is designed for autonomous operation. Update version number when making significant changes.*
