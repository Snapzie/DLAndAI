from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table
import json

# Load the PDF
elements = partition_pdf(filename="./Data/GPTPolicy.pdf")

output = {}
current_section = "Unknown Section"
counter = 0

for el in elements:
    if not el.text:
        continue

    # Update current section if this is a section header or title
    if el.category in ["Title", "Section Header"]:
        current_section = el.text.strip()
        continue

    # Handle tables as Markdown text
    if isinstance(el, Table):
        text = el.to_markdown()
    else:
        text = el.text.strip()

    page = getattr(el.metadata, "page_number", None)

    output[str(counter)] = {
        "text": text,
        "section": current_section,
        "page": page
    }
    counter += 1

# Save to JSON file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("✅ PDF converted to structured JSON as 'output.json'")
