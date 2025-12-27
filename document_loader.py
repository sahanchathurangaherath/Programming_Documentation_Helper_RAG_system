import io
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

def load_pdf(file_obj):
    """
    Extract text from a PDF file object.
    
    Args:
        file_obj: A file-like object containing the PDF data (bytes).
        
    Returns:
        str: The extracted text from the PDF.
    """
    try:
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def load_url(url):
    """
    Extract text from a URL.
    
    Args:
        url (str): The URL to scrape.
        
    Returns:
        str: The extracted text from the webpage.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        raise Exception(f"Error loading URL: {str(e)}")
