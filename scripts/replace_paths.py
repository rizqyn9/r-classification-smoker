import os
import glob

def replace_in_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

# Update smoker scripts
smoker_scripts = glob.glob("scripts/smoker/*.R")
smoker_replacements = [
    ('here("data", "topics", "smoker"', 'here("data", "smoker"'),
    ('here("data", "topics", "extreme_poverty"', 'here("data", "extreme_poverty"')
]

for script in smoker_scripts:
    replace_in_file(script, smoker_replacements)

print("Done updating paths to remove 'topics' nesting.")
