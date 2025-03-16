import csv

# Read valid mesh codes from mesh_terms.csv
with open("data/mesh_terms.csv", mode="r", newline="", encoding="utf-8") as terms_file:
    reader = csv.DictReader(terms_file)
    valid_mesh_codes = {row["DescriptorUI"] for row in reader}

# Read valid UMLS codes from d2g_final_new.csv
with open("data/d2g_final_new.csv", mode="r", newline="", encoding="utf-8") as d2g_file:
    reader = csv.DictReader(d2g_file)
    valid_umls_codes = {row["UMLS"] for row in reader}

# Initialize a set to store UMLS codes where UMLS is valid but mesh_code is missing or invalid
umls_missing_mesh = set()

# Open mesh_umls.csv, filter rows, and write to mesh_umls_final.csv
with open("data/mesh_umls.csv", mode="r", newline="", encoding="utf-8") as umls_file, \
     open("data/mesh_umls_final.csv", mode="w", newline="", encoding="utf-8") as out_file:
    
    reader = csv.DictReader(umls_file)
    writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)
    
    # Write header to the output CSV
    writer.writeheader()
    
    # Process each row
    for row in reader:
        mesh_valid = row["mesh_code"] in valid_mesh_codes
        umls_valid = row["umls_code"] in valid_umls_codes
        
        # Write row only if both mesh_code and umls_code are valid
        if mesh_valid and umls_valid:
            writer.writerow(row)
        # If UMLS code is valid but mesh_code is not, store the UMLS code
        elif umls_valid and not mesh_valid:
            umls_missing_mesh.add(row["umls_code"])

# Print out the UMLS codes that are valid but missing a valid mesh code
if umls_missing_mesh:
    print("UMLS codes with valid UMLS but missing valid mesh code:")
    for code in sorted(umls_missing_mesh):
        print(code)
else:
    print("No UMLS codes found that have a valid UMLS code but missing a valid mesh code.")