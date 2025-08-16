from pdfminer.high_level import extract_text as pdfminer_extract

def extract_text(pdf_path):
    return pdfminer_extract(pdf_path)