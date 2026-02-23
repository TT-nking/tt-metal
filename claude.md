# Claude Development Notes - DeepSeek-V3

## Environment Setup

### Python Environment
Always activate the Python environment before running tests:
```bash
source python_env/bin/activate
```

### Environment Variables for DeepSeek-V3

#### Model Paths
- **Model Directory**: `/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52`
  - Environment variable: `DEEPSEEK_V3_HF_MODEL`

- **Model Cache**: `/tmp/deepseek2`
  - Environment variable: `DEEPSEEK_V3_CACHE`
  - Clear cache if needed: `rm -rf /tmp/deepseek2/*`

#### Device Configuration
- **Mesh Device**: Set to `TG` for testing
  - Environment variable: `MESH_DEVICE`

### Running Tests

Example command to run MLP tests:
```bash
source python_env/bin/activate
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass -xvs --timeout=300
```

Example command to run MoE tests:
```bash
source python_env/bin/activate
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_moe.py::test_forward_pass -xvs --timeout=300
```

## Running Tests

To run the MLP tests:
```bash
source python_env/bin/activate
rm -rf /tmp/deepseek2/*  # Clear cache if needed
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass -xvs --timeout=300
```

To run the MoE tests:
```bash
source python_env/bin/activate
rm -rf /tmp/deepseek2/*  # Clear cache if needed
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_moe.py::test_forward_pass -xvs --timeout=300
```

## Recent Changes

### 2026-02-23: Investigated Collective Operations in Tests
- **Initial attempt**: Tried to remove collective operations from test_mlp.py and test_moe.py per PR review comment
- **Finding**: The collective operations in tests are **NOT redundant** - they are necessary because:
  - MLP and MoE modules have sharded weights for tensor parallelism
  - The modules themselves don't handle collective operations (by design)
  - When decoder blocks call these modules, they handle collective ops in `_forward_mlp_common`
  - When tests call modules directly, the tests MUST handle collective ops
- **Conclusion**: Reverted changes - the original test implementation with collective operations is correct
- **Architecture**:
  - Modules (MLP, MoE) work with sharded weights and expect appropriate tensor distribution
  - Collective ops are handled by the caller (decoder blocks in production, tests when testing directly)
