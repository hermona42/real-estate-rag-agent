import io
import logging
from typing import List, Optional, Union

from pypdf import PdfReader
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from langchain_core.documents import Document

from app.config import settings

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
        cmd = tesseract_cmd or settings.TESSERACT_CMD
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

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

    def _extract_page_with_pdf2image(self, pdf_bytes: bytes, page_num: int) -> str:
        """Render a PDF page to an image and run OCR (for fully scanned documents)."""
        try:
            images = convert_from_bytes(
                pdf_bytes,
                first_page=page_num,
                last_page=page_num,
            )
            if not images:
                return ""
            return pytesseract.image_to_string(images[0]).strip()
        except Exception as e:
            logger.warning(f"pdf2image OCR failed on page {page_num}: {e}")
            return ""

    def process_pdf(self, file_path_or_stream: Union[str, io.BytesIO], source_name: Optional[str] = None) -> List[Document]:
        """
        Extract text from a PDF file. Uses pypdf for digital text and 
        pytesseract as an OCR fallback for scanned pages.

        Args:
            file_path_or_stream: Path to the PDF file or a file-like object containing bytes.
            source_name: Optional name for metadata. Defaults to path or "stream".

        Returns:
            A list of Document objects, one for each page.
        """
        documents = []
        source_display = source_name if source_name else (file_path_or_stream if isinstance(file_path_or_stream, str) else "stream")
        pdf_bytes: Optional[bytes] = None
        if isinstance(file_path_or_stream, io.BytesIO):
            pdf_bytes = file_path_or_stream.getvalue()
            file_path_or_stream.seek(0)
        
        try:
            reader = PdfReader(file_path_or_stream)
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
                        text = "\n\n".join(ocr_text_parts)
                    elif pdf_bytes:
                        text = self._extract_page_with_pdf2image(pdf_bytes, page_num)
                
                metadata = {
                    "source": source_display,
                    "page": page_num,
                    "total_pages": total_pages
                }
                
                doc = Document(page_content=text or "", metadata=metadata)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error processing PDF {source_display}: {e}")
            raise
            
        return documents


def process_pdf_document(contents: bytes, filename: str) -> List[Document]:
    """
    Helper function to process a PDF from memory (bytes) in FastAPI endpoints.
    """
    processor = PDFProcessor()
    return processor.process_pdf(io.BytesIO(contents), source_name=filename)
