import PyPDF2
from docx import Document   # 👈 ye sahi import hai

def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():   # 👈 crash avoid
            text += page.extract_text()
    return text.lower()

def extract_docx(file):
    doc = Document(file)   # 👈 yaha Document use hoga
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text.lower()
