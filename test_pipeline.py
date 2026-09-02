import sys
import requests
from fpdf import FPDF

def create_sample_lease(filename="test_lease.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Lease Details", ln=1, align='C')
    pdf.cell(200, 10, txt="Rent: $2,500/month", ln=2)
    pdf.cell(200, 10, txt="Security Deposit: $5,000", ln=3)
    pdf.cell(200, 10, txt="Pet Fee: $300", ln=4)
    pdf.cell(200, 10, txt="Late Fee: $100 after the 5th", ln=5)
    pdf.output(filename)
    print(f"Generated {filename}")

def test_pipeline():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Check if server is online
    try:
        res = requests.get(f"{base_url}/")
        res.raise_for_status()
        print("Server is online.")
    except requests.exceptions.RequestException:
        print("Error: The FastAPI server at http://127.0.0.1:8000 is offline.")
        print("Please start Uvicorn first (e.g., uvicorn app.main:app --reload).")
        sys.exit(1)
        
    # 2. Generate PDF
    pdf_filename = "test_lease.pdf"
    create_sample_lease(pdf_filename)
    
    # 3. Upload PDF
    print("\nUploading PDF...")
    with open(pdf_filename, "rb") as f:
        files = {"file": (pdf_filename, f, "application/pdf")}
        upload_res = requests.post(f"{base_url}/upload", files=files)
        
    if upload_res.status_code == 200:
        print("Upload successful:", upload_res.json())
    else:
        print(f"Upload failed with status {upload_res.status_code}:", upload_res.text)
        sys.exit(1)
        
    # 4. Chat query
    question = "What is the rent amount and late fee policy?"
    print(f"\nAsking question: '{question}'")
    chat_res = requests.post(
        f"{base_url}/chat", 
        json={"question": question}
    )
    
    # 5. Print answer and sources
    if chat_res.status_code == 200:
        data = chat_res.json()
        print("\n=== Answer ===")
        print(data.get("answer"))
        print("\n=== Sources ===")
        for idx, source in enumerate(data.get("sources", [])):
            print(f"Source {idx + 1}:")
            print(f"  File: {source.get('source')}")
            print(f"  Page: {source.get('page')}")
            print(f"  Snippet: {source.get('snippet')}")
    else:
        print(f"Chat failed with status {chat_res.status_code}:", chat_res.text)
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
