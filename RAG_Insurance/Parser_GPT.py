from unstructured.partition.pdf import partition_pdf
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
    if el.category in ["Title"]:
        current_section = el.text.strip()
        continue
    
    text = el.text.strip()

    page = getattr(el.metadata, "page_number", None)

    output[str(counter)] = {
        "text": text,
        "section": current_section,
        "page": page,
        "name": 'GPT'
    }
    counter += 1

# Save to JSON file
with open("Data/GPTPolicy.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("PDF converted and saved as 'GPTPolicy.json'")
