from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean
# from unstructured.documents.elements import Table
import json

def flush_section(counter, section_text, current_section, current_page, output):
    # Clean all the data
    clean_text = []
    for chunk in section_text:
        cleaned = clean(chunk, bullets=True,lowercase=True,extra_whitespace=True,dashes=True)
        clean_text.append(cleaned)

    # Join together grouped content in json friendly format
    output[str(counter)] = {
        "text": " ".join(clean_text),
        "section": current_section,
        "page": current_page
    }
    counter += 1
    return counter

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

for el in elements:
    if not el.text or el.category in skip_categories:
        continue
    
    # ListItems and items those last acharacter is ':' are appended to the current section
    if el.category == 'ListItem' or el.text.strip()[-1] == ':':
        section_text.append(el.text)
        continue

    # Case for tables could go here

    # New section if Title or NarrativeText
    if el.category in section_starts:
        section_text.append(el.text)
        if el.category == 'Title':
            current_section = el.text.strip()
        current_page = getattr(el.metadata, "page_number", None)
        counter = flush_section(counter, section_text, current_section, current_page, output)
        section_text = []
        continue

    if current_page is None:
        current_page = getattr(el.metadata, "page_number", None)

# Last section
counter = flush_section(counter, section_text, current_section, current_page, output)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("Output saved as 'output.json'")
