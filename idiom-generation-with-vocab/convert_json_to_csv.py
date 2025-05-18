import json
import csv

# Path to input JSON file
json_file_path = 'dataset/tr/idioms_data.json'

# Path to output CSV file
csv_file_path = 'dataset/tr/idioms_data.csv'

# Read JSON data
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Open CSV file for writing
with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
    # Define custom CSV writer with double quote character as quotechar
    csv_writer = csv.writer(csv_file, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write header
    csv_writer.writerow(['id', 'language', 'sentence', 'tokenized_sentence', 'expression', 'category', 'indices'])
    
    # Write data rows
    for item in data:
        # Convert tokenized_sentence to the exact format in the example
        tokens = []
        for token in item['tokenized_sentence']:
            tokens.append(f"'{token}'")
        tokenized_str = f"[{', '.join(tokens)}]"
        
        # Convert indices to the exact format in the example
        if item['indices'] == [-1]:
            indices_str = "[-1]"
        else:
            indices_str = f"[{', '.join(map(str, item['indices']))}]"
        
        # Write row with correct quoting
        csv_writer.writerow([
            item['id'],
            item['language'],
            item['sentence'],
            tokenized_str,
            item['expression'],
            item['category'],
            indices_str
        ])

print(f"Conversion completed. CSV file saved to {csv_file_path}")