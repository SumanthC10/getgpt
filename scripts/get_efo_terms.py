import re
from rdflib import Graph, RDF, RDFS, URIRef, Literal
from rdflib.namespace import OWL, SKOS
from tqdm import tqdm

OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"
HAS_EXACT_SYNONYM = URIRef(OBO_IN_OWL + "hasExactSynonym")
HAS_RELATED_SYNONYM = URIRef(OBO_IN_OWL + "hasRelatedSynonym")


def clean_text(text: str) -> str:
    """Cleans text by removing URLs, brackets, and normalizing whitespace."""
    if not isinstance(text, str): return ""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = ' '.join(text.split())
    return text.strip()


# 1. Load the ontology
print("Parsing ontology file...")
g = Graph().parse("data/efo.owl")


# 2. Helpers
def iri_to_curie(iri: str) -> str:
    if iri.startswith("http://purl.obolibrary.org/obo/"):
        return iri.rsplit("/", 1)[-1].replace("_", ":")
    if iri.startswith("http://www.ebi.ac.uk/efo/"):
        return "EFO:" + iri.rsplit("_", 1)[-1]
    return iri

REPLACED_BY = URIRef("http://purl.obolibrary.org/obo/IAO_0100001")
def resolve_current(cls: URIRef, seen=None):
    if seen is None: seen = set()
    if cls in seen: return cls
    seen.add(cls)
    repl = g.value(cls, REPLACED_BY)
    if repl: return resolve_current(repl, seen)
    return cls


# 3. Walk every class and extract data, including synonyms
IAO_DEF = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
rows = []
all_classes = list(g.subjects(RDF.type, OWL.Class))

for cls in tqdm(all_classes, desc="✅ Processing ontology classes"):
    if not isinstance(cls, URIRef): continue

    canonical = resolve_current(cls)
    if canonical != cls: continue

    curie = iri_to_curie(str(canonical))
    if not (curie.startswith("EFO:") or curie.startswith("MONDO:")): continue

    label = clean_text(g.value(canonical, RDFS.label, default=Literal("")).toPython())
    definition = clean_text((
        g.value(canonical, IAO_DEF) or
        g.value(canonical, URIRef(OBO_IN_OWL + "hasDefinition")) or
        g.value(canonical, SKOS.definition) or
        Literal("")
    ).toPython())

    synonyms = set()
    for synonym_prop in [HAS_EXACT_SYNONYM, HAS_RELATED_SYNONYM, SKOS.altLabel]:
        for s in g.objects(canonical, synonym_prop):
            synonyms.add(clean_text(s.toPython()))
    
    # Combine synonyms into a single string
    synonyms_str = "; ".join(sorted(list(synonyms)))
    
    rows.append((curie, label, definition, synonyms_str))


# 4. Write out the cleaned data with the new synonyms column
output_file = "data/efo_ontology_terms.tsv"
print(f"Writing {len(rows)} terms to file: {output_file}")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("id\tlabel\tdefinition\tsynonyms\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")

print(f"✅ Successfully wrote cleaned terms and synonyms to {output_file}")