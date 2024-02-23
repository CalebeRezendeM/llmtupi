import re
import json
from tqdm import tqdm

def extract_examples(verbetes):
    # Define the refined regex pattern
    regex_pattern = re.compile(r'[:;]\s*([^:-]+)\s+-\s+([^(\n]+)\s*\(([^,]+,[^,]+,[^\)]+)\)')
    
    # Initialize a set to accumulate unique examples
    examples_set = set()
    
    # Iterate through each verbete in the list
    for verbete in tqdm(verbetes):
        # Find all matches of the regex pattern in the current verbete
        matches = regex_pattern.findall(verbete)
        
        # Add each found example to the set (to ensure uniqueness)
        for match in matches:
            # Formatting the match for clarity and adding to the set
            # create a tuple to match  f"Tupi: {match[0].strip()} - Translation: {match[1].strip()} (Citation: {match[2].strip()})"
            if 'IBGE' not in match[0] and ('VLB' in match[2] or 
                                           'Anchieta' in match[2] or 
                                           'Anch' in match[2] or 
                                           'Léry' in match[2] or 
                                           'Ar.' in match[2] or 
                                           'Ar.' in match[2]):
                print(f"Tupi: {match[0].strip()} \tTranslation: {match[1].strip()}\tCitation: ({match[2].strip()})")
                example_tup = (match[0].strip(), match[1].strip(), match[2].strip())

                examples_set.add(example_tup)
    
    # Return the set of unique examples
    return examples_set

# Example usage
verbetes_list = [
    "Verbete example text : Tupi example - Translation text (Citation, with, commas)",
    # Add more verbete strings as needed
]

with open('docs/dict-conjugated.json', 'r') as f:
    verbetes_list = json.load(f)


# Extract examples
examples = extract_examples([x['d'] for x in verbetes_list])

# write examples to file
with open('docs/citations.json', 'w') as f:
    json.dump(list(examples), f)
