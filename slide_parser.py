import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO
from PIL import Image
import json
from dataclasses import dataclass
import fitz  # PyMuPDF
import os
import time
from pathlib import Path

@dataclass
class SectionResult:
    """Represents the parsing result of a slide section"""
    box_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    parsed_content: Dict[str, Any]
    confidence: float


@dataclass
class SlideMetadata:
    """Metadata for a slide from PDF"""
    slide_number: int
    page_number: int  # 0-indexed page number in PDF
    original_size: Tuple[int, int]  # (width, height)
    extraction_dpi: int
    pdf_path: str


@dataclass
class PDFParsingResult:
    """Complete result of parsing a PDF document"""
    pdf_path: str
    total_slides: int
    successful_slides: int
    failed_slides: int
    slides: List[Dict[str, Any]]  # List of parsed slide results
    processing_summary: Dict[str, Any]
    metadata: Dict[str, Any]


class PDFSlideExtractor:
    """
    Class for extracting slides from PDF documents using PyMuPDF (fitz).
    """
    
    def __init__(self, segmentation_dpi: int = 300, low_res_dpi: int = 200, high_res_dpi: int = 1000):
        """
        Initialize the PDF slide extractor.
        
        Args:
            segmentation_dpi: DPI for extracting slides for segmentation
            low_res_dpi: DPI for low resolution overview images
            high_res_dpi: DPI for high resolution section images
        """
        self.segmentation_dpi = segmentation_dpi
        self.low_res_dpi = low_res_dpi
        self.high_res_dpi = high_res_dpi
    
    def extract_slide_image(self, pdf_path: str, page_number: int, dpi: int = 300) -> Tuple[np.ndarray, SlideMetadata]:
        """
        Extract a single slide image from PDF at specified DPI.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract (0-indexed)
            dpi: DPI for image extraction
            
        Returns:
            Tuple of (image as numpy array, slide metadata)
        """
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            if page_number >= len(pdf_document):
                raise ValueError(f"Page {page_number} does not exist. PDF has {len(pdf_document)} pages.")
            
            # Get the page
            page = pdf_document.load_page(page_number)
            
            # Calculate zoom factor for desired DPI
            # fitz default is 72 DPI
            zoom_factor = dpi / 72.0
            matrix = fitz.Matrix(zoom_factor, zoom_factor)
            
            # Render page to image
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(BytesIO(img_data))
            
            # Convert to numpy array (RGB format)
            image_rgb = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV compatibility
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Create metadata
            metadata = SlideMetadata(
                slide_number=page_number + 1,  # 1-indexed for user display
                page_number=page_number,
                original_size=(pix.width, pix.height),
                extraction_dpi=dpi,
                pdf_path=pdf_path
            )
            
            pdf_document.close()
            return image_bgr, metadata
            
        except Exception as e:
            if 'pdf_document' in locals():
                pdf_document.close()
            raise RuntimeError(f"Failed to extract slide {page_number} from {pdf_path}: {str(e)}")
    
    def extract_all_slides(self, pdf_path: str, dpi: int = 300) -> List[Tuple[np.ndarray, SlideMetadata]]:
        """
        Extract all slides from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image extraction
            
        Returns:
            List of tuples (image, metadata) for each slide
        """
        slides = []
        
        try:
            # Open PDF to get page count
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            pdf_document.close()
            
            print(f"Extracting {total_pages} slides from {pdf_path} at {dpi} DPI...")
            
            # Extract each slide
            for page_num in range(total_pages):
                try:
                    image, metadata = self.extract_slide_image(pdf_path, page_num, dpi)
                    slides.append((image, metadata))
                    print(f"  ✓ Extracted slide {page_num + 1}/{total_pages}")
                except Exception as e:
                    print(f"  ✗ Failed to extract slide {page_num + 1}: {str(e)}")
                    continue
            
            print(f"Successfully extracted {len(slides)}/{total_pages} slides")
            return slides
            
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {pdf_path}: {str(e)}")
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get basic information about a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            pdf_document = fitz.open(pdf_path)
            
            info = {
                "file_path": pdf_path,
                "file_size_mb": os.path.getsize(pdf_path) / (1024 * 1024),
                "total_pages": len(pdf_document),
                "metadata": pdf_document.metadata,
                "is_encrypted": pdf_document.needs_pass,
                "page_sizes": []
            }
            
            # Get size info for first few pages
            for i in range(min(3, len(pdf_document))):
                page = pdf_document.load_page(i)
                rect = page.rect
                info["page_sizes"].append({
                    "page": i + 1,
                    "width_points": rect.width,
                    "height_points": rect.height,
                    "width_inches": rect.width / 72,
                    "height_inches": rect.height / 72
                })
            
            pdf_document.close()
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to get PDF info for {pdf_path}: {str(e)}")


class SlideParser:
    """
    Main class for parsing complex slides using segmentation and multimodal LLM calls.
    """
    
    def __init__(self, llm_client, low_res_dpi: int = 200, high_res_dpi: int = 1000):
        """
        Initialize the slide parser.
        
        Args:
            llm_client: The LLM client with invoke method
            low_res_dpi: DPI for the low resolution overview image
            high_res_dpi: DPI for high resolution section images
        """
        self.llm_client = llm_client
        self.low_res_dpi = low_res_dpi
        self.high_res_dpi = high_res_dpi
    
    def resize_image_to_dpi(self, image: np.ndarray, target_dpi: int, original_dpi: int = 300) -> np.ndarray:
        """
        Resize image to target DPI.
        
        Args:
            image: Input image as numpy array
            target_dpi: Target DPI
            original_dpi: Original DPI of the image
            
        Returns:
            Resized image
        """
        scale_factor = target_dpi / original_dpi
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def create_low_res_with_highlighted_box(self, image: np.ndarray, bbox: Tuple[int, int, int, int], original_image_dpi: int = 300) -> np.ndarray:
        """
        Create a low resolution image with one bounding box highlighted.
        
        Args:
            image: Original slide image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Low resolution image with highlighted box
        """
        # Create low resolution version
        low_res_image = self.resize_image_to_dpi(image, self.low_res_dpi)
        
        # Calculate scaling factor for bbox coordinates
        scale_factor = self.low_res_dpi / original_image_dpi
        # Scale bbox coordinates
        x1, y1, x2, y2 = bbox
        scaled_bbox = (
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor),
            int(y2 * scale_factor)
        )
        
        # Draw highlighted rectangle
        highlighted_image = low_res_image.copy()
        cv2.rectangle(highlighted_image, (scaled_bbox[0], scaled_bbox[1]), 
                     (scaled_bbox[2], scaled_bbox[3]), (0, 255, 0), 3)  # Green rectangle
        
        return highlighted_image
    
    def extract_section_high_res(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract a section of the slide in high resolution.
        
        Args:
            image: Original slide image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            High resolution section image
        """
        # First extract the section from original image
        x1, y1, x2, y2 = bbox
        section = image[y1:y2, x1:x2]
        
        # Resize to high resolution
        high_res_section = self.resize_image_to_dpi(section, self.high_res_dpi)
        
        return high_res_section
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert image to base64 string for LLM input.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def create_section_parsing_messages(self, low_res_b64: str, high_res_b64: str, 
                                      section_id: int) -> List[Any]:
        """
        Create messages for LLM to parse a specific section.
        
        Args:
            low_res_b64: Base64 encoded low resolution image with highlighted box
            high_res_b64: Base64 encoded high resolution section image
            section_id: ID of the section being parsed
            
        Returns:
            List of BaseMessage objects for LLM
        """
        from langchain_core.messages import HumanMessage
        
        system_prompt = """You are an expert at parsing slide content. You will be given:
1. A low-resolution overview of a slide with one section highlighted in a green rectangle
2. A high-resolution image of that specific section

Your task is to parse the highlighted section in detail and return a JSON object with the following structure:
{
    "section_type": "text|image|table|chart|diagram|title|bullet_points",
    "content": {
        "text": "extracted text if any",
        "elements": ["list of visual elements"],
        "relationships": "description of how elements relate",
        "key_information": "most important information in this section"
    },
    "position_context": "description of where this section appears relative to the whole slide",
    "confidence": 0.95
}

Focus on accuracy and detail. Extract all text verbatim and describe visual elements precisely."""

        messages = [
            HumanMessage(content=[
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": f"Parse section {section_id}. Here is the slide overview with the section highlighted:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{low_res_b64}"}},
                {"type": "text", "text": "And here is the high-resolution view of the section to parse:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{high_res_b64}"}},
                {"type": "text", "text": "Return only the JSON object with the parsed content."}
            ])
        ]
        
        return messages
    
    def parse_section(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     section_id: int, debug_output_dir: str = None) -> SectionResult:
        """
        Parse a single section of the slide.
        
        Args:
            image: Original slide image
            bbox: Bounding box coordinates
            section_id: ID of the section
            debug_output_dir: Optional directory to save debug images
            
        Returns:
            SectionResult object with parsed content
        """
        try:
            # Create low resolution image with highlighted box
            low_res_highlighted = self.create_low_res_with_highlighted_box(image, bbox)
            
            # Extract high resolution section
            high_res_section = self.extract_section_high_res(image, bbox)
            
            # Debug: Save images for visualization if requested
            if debug_output_dir:
                os.makedirs(debug_output_dir, exist_ok=True)
                
                # Save low resolution highlighted image
                low_res_path = os.path.join(debug_output_dir, f"section_{section_id:03d}_low_res_highlighted.png")
                cv2.imwrite(low_res_path, low_res_highlighted)
                
                # Save high resolution section
                high_res_path = os.path.join(debug_output_dir, f"section_{section_id:03d}_high_res_section.png")
                cv2.imwrite(high_res_path, high_res_section)
                
                # Save original section crop for comparison
                x1, y1, x2, y2 = bbox
                original_section = image[y1:y2, x1:x2]
                original_path = os.path.join(debug_output_dir, f"section_{section_id:03d}_original_crop.png")
                cv2.imwrite(original_path, original_section)
                
                print(f"    Debug images saved for section {section_id}:")
                print(f"      Low-res highlighted: {low_res_path}")
                print(f"      High-res section: {high_res_path}")
                print(f"      Original crop: {original_path}")
            
            # Convert to base64
            low_res_b64 = self.image_to_base64(low_res_highlighted)
            high_res_b64 = self.image_to_base64(high_res_section)
            
            # Create messages for LLM
            messages = self.create_section_parsing_messages(low_res_b64, high_res_b64, section_id)
            
            # Call LLM
            response = self.llm_client.invoke(messages, response_format="json_object")
            
            # Parse response (assuming it returns JSON)
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Try to parse JSON from response
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, wrap in a basic structure
                parsed_content = {
                    "section_type": "unknown",
                    "content": {"text": content},
                    "confidence": 0.5
                }
            
            return SectionResult(
                box_id=section_id,
                bbox=bbox,
                parsed_content=parsed_content,
                confidence=parsed_content.get("confidence", 0.5)
            )
            
        except Exception as e:
            # Return error result
            return SectionResult(
                box_id=section_id,
                bbox=bbox,
                parsed_content={
                    "section_type": "error",
                    "content": {"error": str(e)},
                    "confidence": 0.0
                },
                confidence=0.0
            )
    
    def parse_sections_parallel(self, image: np.ndarray, detections, 
                              debug_output_dir: str = None) -> List[SectionResult]:
        """
        Parse all sections in parallel.
        
        Args:
            image: Original slide image
            detections: Detections object with bounding boxes
            debug_output_dir: Optional directory to save debug images
            
        Returns:
            List of SectionResult objects
        """
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all parsing tasks
            future_to_section = {}
            
            for i, bbox in enumerate(detections.xyxy):
                # Convert bbox to tuple of ints
                bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                
                future = executor.submit(self.parse_section, image, bbox_tuple, i, debug_output_dir)
                future_to_section[future] = i
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_section):
                section_id = future_to_section[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = SectionResult(
                        box_id=section_id,
                        bbox=(0, 0, 0, 0),
                        parsed_content={
                            "section_type": "error",
                            "content": {"error": str(e)},
                            "confidence": 0.0
                        },
                        confidence=0.0
                    )
                    results.append(error_result)
        
        # Sort results by box_id to maintain order
        results.sort(key=lambda x: x.box_id)
        return results
    
    def create_final_aggregation_messages(self, image: np.ndarray, 
                                        section_results: List[SectionResult]) -> List[Any]:
        """
        Create messages for final aggregation LLM call.
        
        Args:
            image: Original slide image
            section_results: List of parsed section results
            
        Returns:
            List of BaseMessage objects for final LLM call
        """
        from langchain_core.messages import HumanMessage
        
        # Convert original image to base64
        original_b64 = self.image_to_base64(image)
        
        # Prepare section summaries
        sections_summary = []
        for result in section_results:
            sections_summary.append({
                "section_id": result.box_id,
                "bbox": result.bbox,
                "parsed_content": result.parsed_content,
                "confidence": result.confidence
            })
        
        system_prompt = """You are an expert at synthesizing slide content. You have been given:
1. The original slide image
2. Detailed parsing results from multiple sections of the slide

Your task is to create a final, complete, and coherent JSON representation of the entire slide that:
- Combines all section information intelligently
- Maintains spatial relationships between sections
- Resolves any conflicts or overlaps between sections
- Provides a comprehensive structure

Return a JSON object with this structure:
{
    "slide_title": "extracted or inferred title",
    "slide_type": "presentation|document|diagram|mixed",
    "layout_structure": "description of overall layout",
    "sections": [
        {
            "section_id": 0,
            "type": "text|image|table|chart|etc",
            "content": "detailed content",
            "importance": "high|medium|low"
        }
    ],
    "relationships": "how sections relate to each other",
    "key_takeaways": ["main points from the slide"],
    "confidence": 0.95
}"""

        messages = [
            HumanMessage(content=[
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": "Here is the original slide:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_b64}"}},
                {"type": "text", "text": f"And here are the detailed parsing results from {len(section_results)} sections:"},
                {"type": "text", "text": json.dumps(sections_summary, indent=2)},
                {"type": "text", "text": "Provide the final comprehensive JSON representation of this slide."}
            ])
        ]
        
        return messages
    
    def aggregate_final_result(self, image: np.ndarray, 
                             section_results: List[SectionResult]) -> Dict[str, Any]:
        """
        Perform final aggregation of all section results.
        
        Args:
            image: Original slide image
            section_results: List of parsed section results
            
        Returns:
            Final aggregated result as dictionary
        """
        try:
            # Create messages for final LLM call
            messages = self.create_final_aggregation_messages(image, section_results)
            
            # Call LLM for final aggregation
            response = self.llm_client.invoke(messages, response_format="json_object")
            
            # Parse response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Try to parse JSON from response
            try:
                final_result = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, create a basic structure
                final_result = {
                    "slide_title": "Parsed Slide",
                    "slide_type": "mixed",
                    "sections": [result.parsed_content for result in section_results],
                    "raw_response": content,
                    "confidence": 0.5
                }
            
            return final_result
            
        except Exception as e:
            # Return error result
            return {
                "slide_title": "Error in Final Aggregation",
                "slide_type": "error",
                "sections": [result.parsed_content for result in section_results],
                "error": str(e),
                "confidence": 0.0
            }
    
    def parse_slide(self, image: np.ndarray, detections, 
                   debug_output_dir: str = None) -> Dict[str, Any]:
        """
        Main method to parse a complete slide.
        
        Args:
            image: Original slide image as numpy array
            detections: Detections object with bounding boxes from segmentation
            debug_output_dir: Optional directory to save debug images
            
        Returns:
            Complete parsed slide as dictionary
        """
        print(f"Starting slide parsing with {len(detections.xyxy)} detected sections...")
        
        if debug_output_dir:
            print(f"Debug mode enabled - images will be saved to: {debug_output_dir}")
        
        # Step 1: Parse all sections in parallel
        print("Parsing individual sections...")
        section_results = self.parse_sections_parallel(image, detections, debug_output_dir)
        
        # Step 2: Aggregate results
        print("Aggregating final result...")
        final_result = self.aggregate_final_result(image, section_results)
        
        # Add metadata
        final_result["metadata"] = {
            "num_sections": len(section_results),
            "processing_method": "segmentation_and_parallel_llm",
            "low_res_dpi": self.low_res_dpi,
            "high_res_dpi": self.high_res_dpi,
            "section_confidences": [r.confidence for r in section_results]
        }
        
        print("Slide parsing completed!")
        return final_result


class PDFSlideParser:
    """
    Comprehensive class for parsing all slides in a PDF document.
    Combines PDF extraction, segmentation, and LLM-based parsing.
    """
    
    def __init__(self, llm_client, segmentation_function, 
                 segmentation_dpi: int = 300, low_res_dpi: int = 200, high_res_dpi: int = 1000):
        """
        Initialize the PDF slide parser.
        
        Args:
            llm_client: The LLM client with invoke method
            segmentation_function: Function that takes an image and returns detections
            segmentation_dpi: DPI for extracting slides for segmentation
            low_res_dpi: DPI for low resolution overview images  
            high_res_dpi: DPI for high resolution section images
        """
        self.llm_client = llm_client
        self.segmentation_function = segmentation_function
        self.pdf_extractor = PDFSlideExtractor(segmentation_dpi, low_res_dpi, high_res_dpi)
        self.slide_parser = SlideParser(llm_client, low_res_dpi, high_res_dpi)
        self.segmentation_dpi = segmentation_dpi
    
    def parse_single_slide_from_pdf(self, pdf_path: str, page_number: int, 
                                  output_dir: str = None, debug_images: bool = False) -> Dict[str, Any]:
        """
        Parse a single slide from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to parse (0-indexed)
            output_dir: Optional directory to save intermediate results
            debug_images: Whether to save debug images for visualization
            
        Returns:
            Parsed slide result
        """
        try:
            # Extract slide image at segmentation DPI
            image, slide_metadata = self.pdf_extractor.extract_slide_image(
                pdf_path, page_number, self.segmentation_dpi
            )
            
            print(f"Processing slide {slide_metadata.slide_number} from {Path(pdf_path).name}...")
            
            # Run segmentation
            print("  Running segmentation...")
            detections = self.segmentation_function(image)
            
            if hasattr(detections, 'xyxy') and len(detections.xyxy) == 0:
                print("  Warning: No regions detected in slide")
                return {
                    "slide_metadata": slide_metadata.__dict__,
                    "error": "No regions detected by segmentation",
                    "parsing_result": None
                }
            
            print(f"  Found {len(detections.xyxy)} regions")
            
            # Setup debug directory if requested
            debug_dir = None
            if debug_images and output_dir:
                debug_dir = os.path.join(output_dir, f"debug_slide_{slide_metadata.slide_number:03d}")
                os.makedirs(debug_dir, exist_ok=True)
                print(f"  Debug images will be saved to: {debug_dir}")
            
            # Parse slide content
            print("  Parsing slide content...")
            parsing_result = self.slide_parser.parse_slide(image, detections, debug_dir)
            
            # Add slide metadata to result
            parsing_result["slide_metadata"] = slide_metadata.__dict__
            parsing_result["pdf_info"] = {
                "source_pdf": pdf_path,
                "page_number": page_number,
                "slide_number": slide_metadata.slide_number
            }
            
            # Save intermediate results if requested
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save slide image
                slide_filename = f"slide_{slide_metadata.slide_number:03d}.png"
                cv2.imwrite(os.path.join(output_dir, slide_filename), image)
                
                # Save parsing result
                json_filename = f"slide_{slide_metadata.slide_number:03d}_parsed.json"
                with open(os.path.join(output_dir, json_filename), 'w', encoding='utf-8') as f:
                    json.dump(parsing_result, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Successfully parsed slide {slide_metadata.slide_number}")
            return parsing_result
            
        except Exception as e:
            error_result = {
                "slide_metadata": {"slide_number": page_number + 1, "page_number": page_number},
                "error": str(e),
                "parsing_result": None
            }
            print(f"  ✗ Failed to parse slide {page_number + 1}: {str(e)}")
            return error_result
    
    def parse_all_slides_in_pdf(self, pdf_path: str, output_dir: str = None, 
                               max_parallel: int = 3, debug_images: bool = False) -> PDFParsingResult:
        """
        Parse all slides in a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional directory to save results
            max_parallel: Maximum number of slides to process in parallel
            debug_images: Whether to save debug images for visualization
            
        Returns:
            Complete PDF parsing result
        """
        start_time = time.time()
        
        try:
            # Get PDF info
            pdf_info = self.pdf_extractor.get_pdf_info(pdf_path)
            total_slides = pdf_info["total_pages"]
            
            print(f"Starting PDF parsing: {Path(pdf_path).name}")
            print(f"Total slides to process: {total_slides}")
            print(f"Parallel processing limit: {max_parallel}")
            print("=" * 60)
            
            # Create output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save PDF info
                with open(os.path.join(output_dir, "pdf_info.json"), 'w') as f:
                    json.dump(pdf_info, f, indent=2)
            
            # Process slides in parallel batches
            all_results = []
            successful_count = 0
            failed_count = 0
            
            # Process in batches to control memory usage
            for batch_start in range(0, total_slides, max_parallel):
                batch_end = min(batch_start + max_parallel, total_slides)
                batch_pages = list(range(batch_start, batch_end))
                
                print(f"\nProcessing batch: slides {batch_start + 1}-{batch_end}")
                
                # Use ThreadPoolExecutor for this batch
                with ThreadPoolExecutor(max_workers=len(batch_pages)) as executor:
                    # Submit all slides in this batch
                    future_to_page = {}
                    for page_num in batch_pages:
                        future = executor.submit(
                            self.parse_single_slide_from_pdf, 
                            pdf_path, page_num, output_dir, debug_images
                        )
                        future_to_page[future] = page_num
                    
                    # Collect results for this batch
                    for future in concurrent.futures.as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                            
                            if result.get("error"):
                                failed_count += 1
                            else:
                                successful_count += 1
                                
                        except Exception as e:
                            failed_count += 1
                            error_result = {
                                "slide_metadata": {"slide_number": page_num + 1, "page_number": page_num},
                                "error": f"Processing exception: {str(e)}",
                                "parsing_result": None
                            }
                            all_results.append(error_result)
            
            # Sort results by slide number
            all_results.sort(key=lambda x: x.get("slide_metadata", {}).get("slide_number", 0))
            
            # Calculate processing time
            total_time = time.time() - start_time
            
            # Create final result
            processing_summary = {
                "total_processing_time_seconds": total_time,
                "average_time_per_slide": total_time / total_slides,
                "successful_slides": successful_count,
                "failed_slides": failed_count,
                "success_rate": successful_count / total_slides,
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = PDFParsingResult(
                pdf_path=pdf_path,
                total_slides=total_slides,
                successful_slides=successful_count,
                failed_slides=failed_count,
                slides=all_results,
                processing_summary=processing_summary,
                metadata=pdf_info
            )
            
            # Save complete result
            if output_dir:
                final_result_path = os.path.join(output_dir, "complete_pdf_parsing_result.json")
                with open(final_result_path, 'w', encoding='utf-8') as f:
                    # Convert dataclass to dict for JSON serialization
                    result_dict = {
                        "pdf_path": result.pdf_path,
                        "total_slides": result.total_slides,
                        "successful_slides": result.successful_slides,
                        "failed_slides": result.failed_slides,
                        "slides": result.slides,
                        "processing_summary": result.processing_summary,
                        "metadata": result.metadata
                    }
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                
                print(f"\nComplete results saved to: {final_result_path}")
            
            # Print summary
            print("\n" + "=" * 60)
            print("PDF PARSING COMPLETED")
            print("=" * 60)
            print(f"PDF: {Path(pdf_path).name}")
            print(f"Total slides: {total_slides}")
            print(f"Successfully parsed: {successful_count}")
            print(f"Failed: {failed_count}")
            print(f"Success rate: {processing_summary['success_rate']:.1%}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average per slide: {processing_summary['average_time_per_slide']:.1f} seconds")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF {pdf_path}: {str(e)}")
    
    def parse_specific_slides(self, pdf_path: str, slide_numbers: List[int], 
                            output_dir: str = None, debug_images: bool = False) -> List[Dict[str, Any]]:
        """
        Parse specific slides from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            slide_numbers: List of slide numbers to parse (1-indexed)
            output_dir: Optional directory to save results
            debug_images: Whether to save debug images for visualization
            
        Returns:
            List of parsed slide results
        """
        results = []
        
        print(f"Parsing {len(slide_numbers)} specific slides from {Path(pdf_path).name}")
        
        for slide_num in slide_numbers:
            page_num = slide_num - 1  # Convert to 0-indexed
            try:
                result = self.parse_single_slide_from_pdf(pdf_path, page_num, output_dir, debug_images)
                results.append(result)
            except Exception as e:
                error_result = {
                    "slide_metadata": {"slide_number": slide_num, "page_number": page_num},
                    "error": str(e),
                    "parsing_result": None
                }
                results.append(error_result)
        
        return results
    

# PDF parsing utility functions
def parse_pdf_slides(pdf_path: str, segmentation_function, llm_client, 
                    output_dir: str = None, segmentation_dpi: int = 300, 
                    max_parallel: int = 3, debug_images: bool = False) -> PDFParsingResult:
    """
    Convenience function to parse all slides in a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        segmentation_function: Function that takes an image and returns detections
        llm_client: Your LLM client
        output_dir: Optional directory to save results
        segmentation_dpi: DPI for extracting slides for segmentation
        max_parallel: Maximum slides to process in parallel
        debug_images: Whether to save debug images for visualization
        
    Returns:
        Complete PDF parsing result
    """
    parser = PDFSlideParser(llm_client, segmentation_function, segmentation_dpi)
    return parser.parse_all_slides_in_pdf(pdf_path, output_dir, max_parallel, debug_images)


def parse_specific_pdf_slides(pdf_path: str, slide_numbers: List[int], 
                             segmentation_function, llm_client, 
                             output_dir: str = None, segmentation_dpi: int = 300,
                             debug_images: bool = False) -> List[Dict[str, Any]]:
    """
    Convenience function to parse specific slides from a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        slide_numbers: List of slide numbers to parse (1-indexed)
        segmentation_function: Function that takes an image and returns detections
        llm_client: Your LLM client
        output_dir: Optional directory to save results
        segmentation_dpi: DPI for extracting slides for segmentation
        debug_images: Whether to save debug images for visualization
        
    Returns:
        List of parsed slide results
    """
    parser = PDFSlideParser(llm_client, segmentation_function, segmentation_dpi)
    return parser.parse_specific_slides(pdf_path, slide_numbers, output_dir, debug_images)


# Utility function to use with your existing segmentation code
def parse_slide_with_detections(image_path: str, merged_detections, llm_client, 
                               output_json_path: str = None, 
                               debug_output_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to parse a slide given the image path and detections.
    
    Args:
        image_path: Path to the slide image
        merged_detections: Detections object from your segmentation code
        llm_client: Your LLM client
        output_json_path: Optional path to save the result JSON
        debug_output_dir: Optional directory to save debug images
        
    Returns:
        Parsed slide result
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create parser and parse
    parser = SlideParser(llm_client)
    result = parser.parse_slide(image, merged_detections, debug_output_dir)
    
    # Save result if path provided
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved to {output_json_path}")
    
    return result


# Example usage
"""
merged_detections = sv.Detections(
    xyxy=merged_boxes,
    confidence=dummy_conf,
    class_id=dummy_class_ids
)

result = parse_slide_with_detections(
    image_path="path/to/your/slide.jpg",
    merged_detections=merged_detections,
    llm_client=your_llm_client,
    output_json_path="parsed_slide.json"
)

print("Parsed slide result:")
print(json.dumps(result, indent=2))
"""
