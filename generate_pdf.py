from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

content = """
RESIDENTIAL LEASE AGREEMENT

1. PROPERTY DETAILS
Address: 742 Evergreen Terrace, Unit 4B, Springfield
Monthly Rent: $2,450 per month, due on the 1st of each month.
Security Deposit: $4,900 required at signing.

2. PET POLICY
No pets are permitted on the premises without prior written approval. 
Approved pets require a non-refundable pet fee of $350 and $50 monthly pet rent.

3. MAINTENANCE AND REPAIRS
The Tenant is responsible for minor repairs under $100. 
The Landlord covers major HVAC, plumbing, and structural maintenance.

4. TERMINATION & RENEWAL
Notice to vacate must be provided in writing at least 60 days prior to lease end.
Late payments after the 5th of the month incur a $150 late fee.
"""

for line in content.strip().split("\n"):
    pdf.cell(200, 10, txt=line, ln=True)

pdf.output("sample_lease.pdf")
print("sample_lease.pdf successfully created!")
