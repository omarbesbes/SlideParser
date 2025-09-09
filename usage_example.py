from slide_parser import parse_pdf_slides, parse_specific_pdf_slides, parse_slide_with_detections
from segment import annotate_slide

llm_client = ... #can be AzureChatOpenAI.client or ChatOpenAI.client

pdf_path = "path/to/your/pdf/document.pdf"
result = parse_pdf_slides(pdf_path=pdf_path, segmentation_function=annotate_slide, llm_client=llm_client)

#or
result = parse_specific_pdf_slides(pdf_path=pdf_path, slide_numbers=[1, 2, 3], segmentation_function=annotate_slide, llm_client=llm_client)

#or
detections = annotate_slide("path/to/your/image.jpg")
result = parse_slide_with_detections(image_path="path/to/your/image.jpg", merged_detections=detections, llm_client=llm_client)