import requests
import nltk
nltk.download('punkt')

# URL of the Flask app
url = "http://127.0.0.1:5002/check_cv"  # Update if running on a different host/port

# File path to the PDF you want to upload
pdf_file_path = "D:/GitHub/Opportune/model/data/ACCOUNTANT/11163645.pdf"  # Replace with the actual path to your PDF file

# Job category and job description
job_category = "ACCOUNTANT"  # Replace with the desired job category
job_description = "We are seeking a detail-oriented and experienced Accountant to join our team. The successful candidate will be responsible for maintaining financial records, preparing reports, and ensuring compliance with financial regulations. This role involves managing accounts payable and receivable, reconciling bank statements, preparing tax returns, and analyzing financial data to assist in decision-making. Additionally, the Accountant will oversee payroll, perform account reconciliations, manage budgets and financial forecasts, and provide financial insights to improve operational efficiency. Proficiency in accounting software, a strong understanding of accounting standards and tax regulations, and excellent analytical and communication skills are essential for success in this role."

# Prepare the files and data for the request
files = {'pdf_file': open(pdf_file_path, 'rb')}
data = {
    'job_category': job_category,
    'job_description': job_description
}

# Send the POST request
response = requests.post(url, files=files, data=data)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Response:", result)
else:
    print("Error:", response.status_code, response.text)
