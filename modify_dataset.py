import json
import random
import re

random.seed(42)

# Read notebook
with open('notebooks/day3_finetuning_ops.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Extract dataset cell (cell 6)
cell = nb['cells'][6]
src = ''.join(cell['source'])

# Find and extract raw_data array using eval (safer than parsing manually)
start = src.find('raw_data =  [')
if start == -1:
    print("Could not find raw_data")
    exit(1)

# Find the matching closing bracket
bracket_count = 0
i = start + len('raw_data =  ')
end = -1

for j in range(i, len(src)):
    if src[j] == '[':
        bracket_count += 1
    elif src[j] == ']':
        bracket_count -= 1
        if bracket_count == 0:
            end = j + 1
            break

if end == -1:
    print("Could not find closing bracket")
    exit(1)

# Extract and evaluate
data_str = src[i:end]
try:
    data = eval(data_str)
    print(f"Loaded {len(data)} examples")
except Exception as e:
    print(f"Error parsing: {e}")
    exit(1)

# Randomly select 40 examples to modify
indices_to_modify = random.sample(range(len(data)), min(40, len(data)))

word_targets = [300, 350, 400, 450, 500]
expansions = [
    " This distinction is critical in high-volume operations where precision directly impacts system performance and regulatory compliance. Organizations that prioritize rigorous data governance in these edge cases consistently outperform competitors in operational metrics and customer satisfaction.",
    " The underlying principle—ensuring data quality at the source—applies across all domains of the travel industry ecosystem. Whether managing ancillary services, regulatory requirements, or cross-system integrations, the same pattern repeats: poor data quality at input creates exponential costs downstream in analytics, reporting, and customer service.",
    " Understanding these nuances requires domain expertise that combines technical knowledge with business acumen. Teams that invest in proper training and system design in these areas see measurable improvements in efficiency, accuracy, and ultimately, customer retention and revenue protection.",
    " This scenario illustrates why investment in proper system design and data governance delivers returns across the entire organization. From finance to operations to customer service, each department benefits when booking data maintains integrity through the entire transaction lifecycle.",
]

modified_count = 0
for idx in indices_to_modify:
    ex = data[idx]
    target = random.choice(word_targets)

    # Modify instruction
    original = ex['instruction']
    ex['instruction'] = f"{original}\n\nProvide a detailed response (approximately {target} words)."

    # Expand response if needed
    current_words = len(ex['output'].split())

    if current_words < target * 0.7:
        ex['output'] += random.choice(expansions)
        modified_count += 1

print(f"Modified {modified_count} examples with expanded responses")

# Save modified data
with open('modified_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Saved to modified_dataset.json")

# Now rebuild the notebook cell
new_cell_src = f"""import json
from datasets import load_dataset

# 1. Provide a small sample of our dataset (In reality, use 50+ examples)
raw_data =  {json.dumps(data, indent=1, ensure_ascii=False)}

dataset = Dataset.from_list(raw_data)
print(f"Dataset size: {{len(dataset)}} examples")
print(f"Example 0: {{dataset[0]['instruction'][:100]}}...")
"""

# Update the notebook cell
nb['cells'][6]['source'] = new_cell_src.split('\n')

# Write back
with open('notebooks/day3_finetuning_ops.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated day3_finetuning_ops.ipynb")
