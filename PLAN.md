# FastVLM Preprocessing Improvement Plan

## Overview
Create a single Python preprocessing script that implements adaptive-resolution and content-aware cropping to optimize FastVLM's input processing.

## Research Gaps Addressed

1. **Adaptive-Resolution Mechanism**: Currently, FastVLM uses fixed resolution regardless of image complexity
2. **Content-Aware Cropping**: Model processes entire image including empty/irrelevant areas

## Script Design: `preprocess_image.py`

### Core Components

#### 1. Image Analysis Module
**Purpose**: Analyze image content to determine complexity

**Features**:
- **Text density detection**: Use OCR or edge detection to identify text-heavy regions
- **Detail level analysis**: Calculate edge density, gradient magnitude, frequency domain analysis
- **Object detection**: Identify informative regions vs empty space
- **Complexity scoring**: Combine metrics into a single complexity score (0-1)

**Methods**:
- Edge detection (Canny, Sobel)
- Variance/entropy calculation for detail assessment
- Optional: lightweight text detection (e.g., EAST detector, tesseract)
- Color histogram analysis for uniformity

#### 2. Adaptive Resolution Selector
**Purpose**: Choose optimal resolution based on complexity score

**Strategy**:
- **Low complexity (score < 0.3)**: Reduce resolution to 224x224 or 336x336
- **Medium complexity (0.3 ≤ score < 0.7)**: Standard resolution 512x512
- **High complexity (score ≥ 0.7)**: Increase resolution to 768x768 or 1024x1024

**Configuration**:
- Configurable resolution tiers
- Threshold adjustments based on use case
- Aspect ratio preservation

#### 3. Content-Aware Cropping Module
**Purpose**: Remove uninformative regions before encoding

**Features**:
- **Background removal**: Detect and crop uniform/empty backgrounds
- **Margin detection**: Remove document margins, letterboxing
- **Bounding box extraction**: Focus on content-rich regions
- **Smart padding**: Add minimal padding around important content

**Techniques**:
- Thresholding for background detection
- Contour detection for content boundaries
- Saliency map generation (optional)
- Connected component analysis

#### 4. Integration Interface
**Purpose**: Seamless integration with existing FastVLM pipeline

**Functions**:
- `preprocess_image(image_path, config)`: Main entry point
- `analyze_complexity(image)`: Returns complexity metrics
- `select_resolution(complexity_score, config)`: Returns target resolution
- `crop_content(image, config)`: Returns cropped image
- `apply_preprocessing(image, config)`: Complete pipeline

## Implementation Steps

### Phase 1: Basic Structure (Week 1)
1. Create base script with CLI interface
2. Implement image loading and basic resizing
3. Add configuration file support (YAML/JSON)
4. Setup logging and visualization options

### Phase 2: Complexity Analysis (Week 2)
1. Implement edge detection metrics
2. Add entropy/variance calculations
3. Create complexity scoring function
4. Test on diverse image dataset
5. Fine-tune thresholds

### Phase 3: Adaptive Resolution (Week 3)
1. Implement resolution selection logic
2. Add aspect ratio preservation
3. Create resolution tiers configuration
4. Benchmark performance impact
5. Compare accuracy vs speed trade-offs

### Phase 4: Content-Aware Cropping (Week 4)
1. Implement background detection
2. Add margin removal algorithm
3. Create smart cropping with padding
4. Handle edge cases (full-content images)
5. Validate on document/natural images

### Phase 5: Integration & Testing (Week 5)
1. Integrate with FastVLM model pipeline
2. Create batch processing support
3. Add visualization mode for debugging
4. Performance benchmarking suite
5. Documentation and examples

## Technical Specifications

### Dependencies
```
opencv-python
numpy
pillow
pyyaml (for config)
scikit-image (optional, for advanced analysis)
pytesseract (optional, for text detection)
```

### Configuration File (`preprocess_config.yaml`)
```yaml
complexity_analysis:
  use_edge_detection: true
  use_entropy: true
  use_text_detection: false  # optional
  
resolution_tiers:
  low: 336
  medium: 512
  high: 768
  ultra: 1024
  
thresholds:
  low_complexity: 0.3
  high_complexity: 0.7
  
cropping:
  enabled: true
  min_margin: 10
  max_crop_ratio: 0.3  # don't crop more than 30% total
  background_threshold: 15  # pixel intensity threshold
  
output:
  save_visualizations: false
  save_metrics: true
  verbose: false
```

### Input/Output Interface
**Input**: 
- Image path or PIL Image object
- Configuration dict or file path

**Output**:
- Preprocessed PIL Image
- Metadata dict with:
  - Original dimensions
  - Final dimensions
  - Complexity score
  - Crop coordinates
  - Selected resolution tier
  - Processing time

## Expected Benefits

1. **Performance**: 20-40% faster inference on simple images
2. **Accuracy**: Better performance on complex/text-heavy images
3. **Efficiency**: Reduced token count by removing uninformative regions
4. **Flexibility**: Configurable trade-offs between speed and accuracy

## Validation Metrics

- **Speed**: Measure inference time before/after preprocessing
- **Accuracy**: Evaluate on benchmark datasets (VQA, DocVQA, etc.)
- **Token Reduction**: Count visual tokens saved via cropping
- **Quality**: Visual inspection of preprocessing results

## Future Extensions

1. Learned complexity scoring (neural network-based)
2. Multi-scale processing for different image regions
3. Integration with attention mechanisms
4. Dynamic tiling based on content distribution
5. GPU acceleration for preprocessing
