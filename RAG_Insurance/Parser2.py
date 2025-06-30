from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table
from unstructured.cleaners.core import clean
import json

def flush_section(counter, section_text, current_section, current_page, output):
    # Split and clean lines on newline
    clean_text = []
    for chunk in section_text:
        cleaned = clean(chunk, bullets=True,lowercase=True,extra_whitespace=True,dashes=True)
        clean_text.append(cleaned)

    output[str(counter)] = {
        "text": " ".join(clean_text),
        "section": current_section,
        "page": current_page
    }
    counter += 1
    return counter

# Load the PDF and extract elements
elements = partition_pdf(filename="./Data/guidlines_risk_management_singapore.pdf",strategy='hi_res',include_page_breaks=False)

# Initialize state
output = {}
section_text = []
current_section = "Beginning of document"
current_page = None
counter = 0

# Allowed categories for content grouping
skip_categories = {"Header", "UncategorizedText"}
section_starts = {"Title", "NarrativeText"}

# Process elements
for el in elements:
    if not el.text or el.category in skip_categories:
        continue
    
    # ListItems and items those last acharacter is ':' are appended to the current section
    if el.category == 'ListItem' or el.text.strip()[-1] == ':':
        section_text.append(el.text)
        continue

    # New section if Title or NarrativeText
    if el.category in section_starts:
        section_text.append(el.text)
        if el.category == 'Title':
            current_section = el.text.strip()
        current_page = getattr(el.metadata, "page_number", None)
        counter = flush_section(counter, section_text, current_section, current_page, output)
        section_text = []
        continue

    # Include ListItem, NarrativeText, Table in current section
    # if isinstance(el, Table):
    #     section_text.append(el.to_markdown())
    # else:


    if current_page is None:
        current_page = getattr(el.metadata, "page_number", None)

# Final flush
counter = flush_section(counter, section_text, current_section, current_page, output)

# Save to file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("✅ Refined PDF content grouped by section saved as 'output.json'")
