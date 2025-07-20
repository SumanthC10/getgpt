from rdflib import Graph, RDF, RDFS, URIRef, Literal
from rdflib.namespace import OWL
from tqdm import tqdm

# 1. Load the ontology -------------------------------------------------
print("Parsing ontology file")
# Ensure the path to your efo.owl file is correct
# ontology file downloaded from https://github.com/EBISPOT/efo/releases
g = Graph().parse("data/efo.owl")

# 2. Helper → CURIE from *any* OBO-style IRI ---------------------------
def iri_to_curie(iri: str) -> str:
    """
    http://purl.obolibrary.org/obo/MONDO_0004947 → MONDO:0004947
    http://www.ebi.ac.uk/efo/EFO_0000094          → EFO:0000094
    Falls back to the full IRI if it doesn’t match the common patterns.
    """
    if iri.startswith("http://purl.obolibrary.org/obo/"):
        return iri.rsplit("/", 1)[-1].replace("_", ":")
    if iri.startswith("http://www.ebi.ac.uk/efo/"):
        return "EFO:" + iri.rsplit("_", 1)[-1]
    return iri

# 3. Helper → chase 'replaced_by' recursively --------------------------
REPLACED_BY = URIRef("http://purl.obolibrary.org/obo/IAO_0100001")

def resolve_current(cls: URIRef, seen=None):
    if seen is None: seen = set()
    if cls in seen:  return cls            # circular safety
    seen.add(cls)

    repl = g.value(cls, REPLACED_BY)
    if repl: return resolve_current(repl, seen)

    return cls

# 4. Walk every class --------------------------------------------------
IAO_DEF = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")

rows = []

# --- Convert generator to a list for tqdm ---
all_classes = list(g.subjects(RDF.type, OWL.Class))

# --- Wrap the iterable with tqdm() for a progress bar ---
for cls in tqdm(all_classes, desc="✅ Processing ontology classes"):
    # Skip any non-IRI subjects (i.e., blank nodes)
    if not isinstance(cls, URIRef):
        continue

    canonical = resolve_current(cls)
    if canonical != cls:
        # skip the obsolete original; the live term will be handled elsewhere
        continue

    curie = iri_to_curie(str(canonical))

    # Only include EFO or MONDO terms
    if not (curie.startswith("EFO:") or curie.startswith("MONDO:")):
        continue

    label = g.value(canonical, RDFS.label, default=Literal("")).toPython()
    definition = (
        g.value(canonical, IAO_DEF)
        or g.value(canonical, URIRef("http://www.geneontology.org/formats/oboInOwl#hasDefinition"))
        or g.value(canonical, URIRef("http://www.w3.org/2004/02/skos/core#definition"))
        or Literal("")
    ).toPython()

    rows.append((curie, label, definition))

# 5. Write out ---------------------------------------------------------
print(f"Writing {len(rows)} terms to file...")
with open("data/efo_ontology_terms.tsv", "w", encoding="utf-8") as f:
    f.write("id\tlabel\tdefinition\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")

print("✅ Successfully wrote EFO and MONDO terms to data/efo_ontology_terms.tsv")