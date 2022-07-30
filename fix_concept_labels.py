import re
import glob
from pathlib import Path

def slug(text):
  text = text.replace("(", " ")
  text = text.replace(")", " ")
  text = re.sub(r"\s+", " ", text).strip()
  text = text.replace(" ", "_")
  
  return text

def main():
  # Load list of concepts and make a dictionary
  with open("tupi_concept_map.txt") as h:
    concepts = h.readlines()
    concepts = [l.strip() for l in concepts]
    concepts = [l for l in concepts if l]
    mapper = {concept:slug(concept) for concept in concepts}   
    
    
  # map all files
  files = glob.glob("new_tupi/*")
  for filename in files:
    with open(filename, encoding="utf-8") as h:
      source = h.read()
      
    dest = source
    for k, v in mapper.items():
      dest = dest.replace(k, v)

    f = Path(filename).parts[-1]
    output = Path(__file__).parent / "saturday" / f    
    with open(output, "w", encoding="utf-8") as h:
      h.write(dest)
        
if __name__ == "__main__":
  main()
