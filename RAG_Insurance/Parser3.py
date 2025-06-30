from unstructured.partition.pdf import partition_pdf
import json

chunks = partition_pdf(
    filename="./Data/guidlines_risk_management_singapore.pdf",
    strategy='hi_res'
)
for chunk in chunks:
    print('-'*40)
    print(chunk.category,chunk.text)

