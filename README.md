# SlideParser

A comprehensive Python library for parsing and analyzing complex presentation slides using computer vision and large language models (LLMs). SlideParser combines YOLO-based object detection, advanced image processing, and multimodal LLMs to extract structured information from PDF presentations and slide images.

## Features

- **PDF Slide Extraction**: Extract individual slides from PDF presentations at configurable DPI
- **Intelligent Segmentation**: Fine-tuned YOLOv11 model for detecting slide components (text, tables, figures, etc.)
- **Color Enhancement**: Automated color distinctification pipeline for improved visual analysis
- **Parallel Processing**: Multi-threaded parsing for efficient batch processing
- **LLM Integration**: Compatible with OpenAI GPT and Anthropic Claude models
- **Structured Output**: JSON-formatted extraction results with metadata and confidence scores

## Installation

## Weights

I used the weights from the following repository:
https://github.com/moured/YOLOv11-Document-Layout-Analysis/

Here is the link to download the weights directly: https://github.com/moured/YOLOv11-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov11x_best.pt


### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### LLM Provider Setup

Choose and install your preferred LLM provider:

**For OpenAI:**
```bash
pip install openai>=1.0.0
```

**For Anthropic:**
```bash
pip install anthropic>=0.25.0
```

## Quick Start

### Basic Usage

```python
from slide_parser import parse_pdf_slides, parse_specific_pdf_slides, parse_slide_with_detections
from segment import annotate_slide

# Initialize your LLM client (example with OpenAI)
from openai import OpenAI
llm_client = OpenAI(api_key="your-api-key")

# Parse all slides in a PDF
pdf_path = "path/to/your/presentation.pdf"
result = parse_pdf_slides(
    pdf_path=pdf_path, 
    segmentation_function=annotate_slide, 
    llm_client=llm_client
)

print(f"Processed {result.successful_slides}/{result.total_slides} slides")
```

### Parse Specific Slides

```python
# Parse only slides 1, 2, and 3
result = parse_specific_pdf_slides(
    pdf_path=pdf_path, 
    slide_numbers=[1, 2, 3], 
    segmentation_function=annotate_slide, 
    llm_client=llm_client
)
```

### Parse Individual Images

```python
# Parse a single slide image
detections = annotate_slide("path/to/slide.jpg")
result = parse_slide_with_detections(
    image_path="path/to/slide.jpg",
    merged_detections=detections,
    llm_client=llm_client,
    output_json_path="parsed_result.json"
)
```

## Architecture

### Core Components

1. **PDFSlideExtractor**: Extracts slides from PDF documents using PyMuPDF
2. **SlideParser**: Main parsing engine with LLM integration
3. **Segmentation Module**: YOLOv11-based object detection for slide components
4. **Color Enhancement**: Automated color distinctification for improved analysis

### Supported Slide Elements

The YOLOv11 model is fine-tuned to detect:
- **Caption**: Image and figure captions
- **Footnote**: Reference notes and citations
- **Formula**: Mathematical equations and expressions
- **List-item**: Bulleted and numbered lists
- **Page-footer**: Page numbering and footer content
- **Page-header**: Header content and titles
- **Picture**: Images, charts, and graphics
- **Section-header**: Section titles and headings
- **Table**: Data tables and structured content
- **Text**: General text blocks
- **Title**: Slide titles and main headings

## Advanced Configuration

### Custom DPI Settings

```python
# High-resolution extraction
result = parse_pdf_slides(
    pdf_path=pdf_path,
    segmentation_function=annotate_slide,
    llm_client=llm_client,
    segmentation_dpi=600,  # Higher DPI for better quality
    debug_images=True      # Save debug images
)
```

### Parallel Processing

```python
# Control parallel processing
result = parse_pdf_slides(
    pdf_path=pdf_path,
    segmentation_function=annotate_slide,
    llm_client=llm_client,
    max_parallel=5  # Process 5 slides simultaneously
)
```

### Color Enhancement

```python
from color import pick_best_variant_and_save

# Enhance slide colors for better analysis
input_image = "slide.png"
enhanced_variants = pick_best_variant_and_save(
    input_image, 
    out_dir='enhanced_slides',
    k=6  # Number of color clusters
)
```

## Output Format

The parser returns structured JSON with the following format:

```json
{
  "slide_metadata": {
    "slide_number": 1,
    "original_size": [1920, 1080],
    "extraction_dpi": 300,
    "pdf_path": "presentation.pdf"
  },
  "sections": [
    {
      "box_id": 0,
      "bbox": [100, 50, 800, 200],
      "parsed_content": {
        "type": "title",
        "text": "Slide Title",
        "confidence": 0.95
      }
    }
  ],
  "summary": {
    "total_sections": 5,
    "processing_time": 2.3,
    "confidence_average": 0.87
  }
}
```

## Model Files

The project includes a fine-tuned YOLOv11 model (`yolov11x_best.pt`) specifically trained for slide component detection. This model provides:

- High accuracy for presentation-specific elements
- Optimized bounding box detection
- Support for complex slide layouts
- Robust handling of various presentation styles

## Performance Considerations

### Memory Usage
- Processing high-DPI images requires significant memory
- Consider reducing DPI for large batch processing
- Use parallel processing limits to prevent memory overflow

### Processing Speed
- Average processing time: 3-5 seconds per slide
- Factors affecting speed:
  - Slide complexity
  - Image resolution
  - LLM response time
  - Number of detected elements

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**YOLO Model Loading:**
```python
# Verify model file exists
import os
assert os.path.exists("yolov11x_best.pt"), "YOLO model file not found"
```

**LLM Connection:**
```python
# Test LLM connectivity
try:
    response = llm_client.invoke([{"role": "user", "content": "Hello"}])
    print("LLM connection successful")
except Exception as e:
    print(f"LLM connection failed: {e}")
```

### Debug Mode

Enable debug mode to save intermediate processing images:

```python
result = parse_pdf_slides(
    pdf_path=pdf_path,
    segmentation_function=annotate_slide,
    llm_client=llm_client,
    debug_images=True,
    output_dir="debug_output"
)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source. Please ensure you comply with the licenses of all dependencies, particularly:
- OpenCV (Apache 2.0)
- Ultralytics YOLO (AGPL-3.0)
- PyMuPDF (AGPL-3.0)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{slideparser2024,
  title={SlideParser: Intelligent Slide Analysis with Computer Vision and LLMs},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/SlideParser}
}
```