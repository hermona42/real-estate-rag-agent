import io
import logging
from typing import List, Optional

from pypdf import PdfReader
import pytesseract
from PIL import Image
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    A processor for extracting text from PDF files, combining digital text extraction
    with OCR fallback for scanned pages.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the PDFProcessor.
        
        Args:
            tesseract_cmd: Optional path to the tesseract executable. 
                           Useful for Windows where tesseract might not be in PATH
                           (e.g., r'C:\Program Files\Tesseract-OCR\tesseract.exe').
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _extract_text_with_ocr(self, image_data: bytes) -> str:
        """
        Fallback OCR extraction for an image extracted from a scanned page.
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed on image: {e}")
            return ""

    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Extract text from a PDF file. Uses pypdf for digital text and 
        pytesseract as an OCR fallback for scanned pages.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of Document objects, one for each page.
        """
        documents = []
        
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                page_num = i + 1
                
                # Attempt standard text extraction first
                text = page.extract_text()
                
                # If extract_text() returns no text or very little text, 
                # we assume it might be a scanned page and fallback to OCR on its images.
                if not text or len(text.strip()) < 50:
                    ocr_text_parts = []
                    
                    # Extract embedded images from the page for OCR
                    if hasattr(page, 'images'):
                        for image_file_object in page.images:
                            ocr_text = self._extract_text_with_ocr(image_file_object.data)
                            if ocr_text:
                                ocr_text_parts.append(ocr_text)
                    
                    if ocr_text_parts:
                        # Combine OCR text from all images on the page
                        text = "\n\n".join(ocr_text_parts)
                
                metadata = {
                    "source": file_path,
                    "page": page_num,
                    "total_pages": total_pages
                }
                
                doc = Document(page_content=text or "", metadata=metadata)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
            
        return documents
