# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### macOS with Metal (GPU)
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME=MPS .. && make -j$(sysctl -n hw.logicalcpu)
```

### macOS CPU-only
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(sysctl -n hw.logicalcpu)
```

### Linux with CUDA
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
```

### Linux with ROCm (AMD GPU)
```bash
mkdir build && cd build
export PYTORCH_ROCM_ARCH=gfx906  # Adjust for your GPU
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME="HIP" -DHIP_ROOT_DIR=/opt/rocm -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON ..
make
```

### Windows
```bash
"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
md build
cd build
cmake -DCMAKE_PREFIX_PATH=C:/path_to/libtorch -DOPENCV_DIR=C:/path_to/OpenCV/build -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Running the Application

### Basic usage
```bash
./build/opensplat /path/to/dataset -n 2000
```

### Common parameters
- `-n, --num-iters`: Number of training iterations (default: 30000)
- `-o, --output`: Output file path (default: splat.ply)
- `-d, --downscale-factor`: Scale input images by factor (default: 1)
- `--resume`: Resume training from a PLY file
- `--val`: Withhold a camera for validation
- `--cpu`: Force CPU execution (no GPU)

### Example with validation
```bash
./build/opensplat /path/to/banana --val --val-render ./validation_renders -n 5000
```

### Generate compressed .splat files
```bash
./build/opensplat /path/to/banana -o banana.splat
```

## Development

### Clean rebuild
```bash
rm -rf build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
```

### View help and all parameters
```bash
./build/opensplat --help
```

## Architecture Overview

### Core Training Pipeline

The training loop is in `opensplat.cpp:main()`:
1. Load input data (COLMAP/OpenSfM/ODM/OpenMVG/nerfstudio format) via `inputDataFromX()`
2. Initialize `Model` with camera parameters and training hyperparameters
3. For each iteration:
   - Select random camera from training set
   - `model.forward()` - render the scene from camera viewpoint
   - Compute loss (L1 + SSIM)
   - Backpropagate gradients
   - Update gaussian parameters via Adam optimizers
   - `model.afterTrain()` - densification/pruning every `refineEvery` steps
4. Save final `.ply` or `.splat` output

### Model Class (model.hpp, model.cpp)

The `Model` class maintains learnable gaussian parameters:
- `means` - 3D positions (Nx3)
- `scales` - Size/scale per axis (Nx3)
- `quats` - Rotation quaternions (Nx4)
- `featuresDc` - Base spherical harmonics color (NxDx3)
- `featuresRest` - Higher-order SH coefficients
- `opacities` - Opacity values (Nx1)

Each parameter has its own Adam optimizer with custom learning rates and schedulers.

Key methods:
- `forward()` - Renders scene from camera view using gaussian splatting
- `afterTrain()` - Handles gaussian densification (split/duplicate) and pruning
- `save()` / `savePly()` / `saveSplat()` - Serialize trained model

### GPU Backend Abstraction

The rasterizer has three implementations selected via `GPU_RUNTIME` CMake variable:

1. **CUDA** (`rasterizer/gsplat/*.cu`) - NVIDIA GPUs
2. **HIP** (`rasterizer/gsplat/*.cu` with `USE_HIP`) - AMD GPUs via ROCm
3. **Metal** (`rasterizer/gsplat-metal/*.metal`, `*.mm`) - Apple Silicon/Metal
4. **CPU** (`rasterizer/gsplat-cpu/*.cpp`) - Fallback CPU implementation

The abstraction layer is in `gsplat.hpp` which includes the appropriate backend headers.

### Forward Pass (project_gaussians.cpp, rasterize_gaussians.cpp)

1. **Projection** (`ProjectGaussians::forward()`):
   - Projects 3D gaussians to 2D screen space
   - Computes 2D covariances and screen-space radii
   - Determines which tiles each gaussian overlaps

2. **Binning** (`binAndSortGaussians()`):
   - Maps gaussians to screen tiles
   - Sorts by tile and depth for efficient rasterization

3. **Rasterization** (`RasterizeGaussians::forward()`):
   - Alpha-composites gaussians per pixel
   - Uses custom CUDA/HIP/Metal kernels for parallelism

Both use PyTorch's Autograd via custom `torch::autograd::Function` subclasses.

### Input Data Loaders

OpenSplat supports multiple SfM/MVS project formats:

- `colmap.cpp` - COLMAP format (cameras.bin, images.bin, points3D.bin)
- `nerfstudio.cpp` - Nerfstudio transforms.json
- `opensfm.cpp` - OpenSfM reconstruction.json
- `openmvg.cpp` - OpenMVG sfm_data.json

All converge to the unified `InputData` structure (`input_data.hpp`):
- `cameras` - Camera intrinsics, extrinsics, image paths
- `points` - Initial 3D point cloud (xyz + rgb)
- `scale` / `translation` - Normalization transform

The loader auto-detects format in `input_data.cpp:inputDataFromX()`.

### Gaussian Refinement Strategy

During training (`model.cpp:afterTrain()`), gaussians are refined every `refineEvery` steps:

1. Track per-gaussian metrics:
   - `xysGradNorm` - Positional gradient magnitude (indicates need for densification)
   - `visCounts` - Visibility frequency
   - `max2DSize` - Screen-space size

2. **Split large gaussians** if:
   - Gradient norm > `densifyGradThresh`
   - Scale > `densifySizeThresh`
   - Screen size > `splitScreenSize` (before `stopScreenSizeAt`)

3. **Duplicate small gaussians** if:
   - Gradient norm > `densifyGradThresh`
   - Scale < `densifySizeThresh`

4. **Prune** gaussians if:
   - Opacity < threshold
   - Screen size too large
   - World-space scale too large

5. **Reset opacity** every `resetAlphaEvery` refinements to avoid dead gaussians

### CMake Build System

Key CMake options (CMakeLists.txt):
- `GPU_RUNTIME` - "CUDA" | "HIP" | "MPS" | "CPU" (default: CUDA)
- `OPENSPLAT_MAX_CUDA_COMPATIBILITY` - Build for all CUDA architectures
- `OPENSPLAT_BUILD_SIMPLE_TRAINER` - Build simplified trainer
- `OPENSPLAT_BUILD_VISUALIZER` - Build real-time visualization (requires Pangolin)
- `OPENSPLAT_USE_FAST_MATH` - Enable fast math optimizations
- `CMAKE_CUDA_ARCHITECTURES` - Target CUDA compute capabilities (default: 70;75;80)

Dependencies fetched via FetchContent:
- nlohmann_json - JSON parsing
- nanoflann - KD-tree for nearest neighbors
- cxxopts - Command-line parsing
- glm - Math library for CUDA/HIP

System dependencies:
- libtorch (PyTorch C++ API)
- OpenCV

### Coordinate System

By default, OpenSplat normalizes input coordinates to center and scale the scene. Use `--keep-crs` flag to retain the original coordinate reference system from the input project.

Transforms are stored in:
- `InputData::scale` - Uniform scale factor
- `InputData::translation` - Translation vector (3D)
- `Camera::camToWorld` - Camera-to-world transform matrix (4x4)

### Output Formats

- **PLY format** (`.ply`): Standard 3D gaussian splatting format with full precision
- **SPLAT format** (`.splat`): Compressed format for web viewers
- **cameras.json**: Camera parameters for some viewers

Output files are saved in the same directory as the output scene file.
