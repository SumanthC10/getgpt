import xml.etree.ElementTree as ET
import csv

def shorten_mesh_file(input_xml_path, output_csv_path):
    """
    Creates a CSV file with three columns:
      - DescriptorUI: The unique ID.
      - PreferredTerm: The preferred term from DescriptorName.
      - SearchTerms: A composite of all terms (preferred + alternative) gathered 
        from all Concept/Term elements.
    """
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header with the new SearchTerms column
        writer.writerow(["DescriptorUI", "PreferredTerm", "SearchTerms"])
        
        # Iterate over each DescriptorRecord and extract fields
        for descriptor in root.findall(".//DescriptorRecord"):
            descriptor_ui = descriptor.findtext("DescriptorUI")
            preferred_term = descriptor.find("DescriptorName").findtext("String")
            # Gather all terms from all Concept/TermList elements
            terms_set = set()
            if preferred_term:
                terms_set.add(preferred_term.strip())
            for concept in descriptor.findall("ConceptList/Concept"):
                for term in concept.findall("TermList/Term"):
                    term_str = term.findtext("String")
                    if term_str:
                        terms_set.add(term_str.strip())
            # Join all collected terms into a composite field (delimited by semicolons)
            search_terms = "; ".join(sorted(terms_set))
            if descriptor_ui and preferred_term:
                writer.writerow([descriptor_ui, preferred_term, search_terms])
                
    print(f"MeSH term file saved to {output_csv_path}")

# Example usage:
input_xml = input("Input MeSH .xml file path: ")
output_csv = "mesh_terms.csv"
shorten_mesh_file(input_xml, output_csv)