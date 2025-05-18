import json
import os

def fix_data():
    print("Starting data fix process...")
    
    # Update paths to use absolute paths
    idioms_data_path = 'C:/Users/ulas_/OneDrive/Masaüstü/idiom-generation/dataset/tr/idioms_data.json'
    tr_idioms_path = 'C:/Users/ulas_/OneDrive/Masaüstü/idiom-generation/idioms/tr-idioms.json'
    
    # Load the idioms data
    with open(idioms_data_path, 'r', encoding='utf-8') as f:
        idioms_data = json.load(f)
    
    # Load the reference expressions
    with open(tr_idioms_path, 'r', encoding='utf-8') as f:
        tr_idioms = json.load(f)
    
    # Get all correct expressions from tr-idioms.json
    correct_expressions = list(tr_idioms.keys())
    
    # Counter for tracking changes
    indices_fixed = 0
    expressions_fixed = 0
    
    # Create updated data
    updated_data = []
    
    # Track which corrections were applied
    corrections_applied = []
    
    # Group data by blocks of 240 items (each group should correspond to one expression)
    total_items = len(idioms_data)
    num_expressions = total_items // 240
    
    print(f"Total items: {total_items}")
    print(f"Number of expressions (groups of 240): {num_expressions}")
    
    for i in range(num_expressions):
        start_idx = i * 240
        end_idx = start_idx + 240
        
        # Get the group of 240 items
        group = idioms_data[start_idx:end_idx]
        
        # Find the correct expression from tr-idioms.json
        # We'll use the first item's expression as reference
        current_expression = group[0]['expression']
        
        # Find the matching expression from tr-idioms.json
        correct_expression = None
        
        # Direct match (ideal case)
        if current_expression in correct_expressions:
            correct_expression = current_expression
        else:
            # Try to find the correct expression by matching the order in tr-idioms.json
            # Assuming the order of expressions in the dataset matches tr-idioms.json
            if i < len(correct_expressions):
                correct_expression = correct_expressions[i]
                print(f"Group {i+1}: Replacing '{current_expression}' with '{correct_expression}'")
        
        # If still not found, try to guess based on similarity
        if not correct_expression:
            for expr in correct_expressions:
                if expr.split()[0] in current_expression or current_expression.split()[0] in expr:
                    correct_expression = expr
                    print(f"Group {i+1}: Guessing match - Replacing '{current_expression}' with '{correct_expression}'")
                    break
        
        # If still no match found, keep the original
        if not correct_expression:
            correct_expression = current_expression
            print(f"Group {i+1}: No match found for '{current_expression}'")
        
        # Update all 240 items in this group
        for item in group:
            # Fix the expression if it doesn't match the correct one
            if item['expression'] != correct_expression:
                old_expression = item['expression']
                item['expression'] = correct_expression
                expressions_fixed += 1
                corrections_applied.append({
                    "id": item['id'],
                    "fix_type": "expression",
                    "before": old_expression,
                    "after": correct_expression
                })
            
            # Fix indices for expressions with more than 2 words
            if len(correct_expression.split()) > 2 and len(item['indices']) == 2:
                start_token_idx = item['indices'][0]
                end_token_idx = item['indices'][1]
                
                # Check if indices are missing intermediate values
                expected_range = list(range(start_token_idx, end_token_idx + 1))
                if len(expected_range) > 2:
                    old_indices = item['indices'].copy()
                    item['indices'] = expected_range
                    indices_fixed += 1
                    corrections_applied.append({
                        "id": item['id'],
                        "fix_type": "indices",
                        "before": old_indices,
                        "after": expected_range
                    })
            
            updated_data.append(item)
    
    # Handle any remaining items (if the total isn't evenly divisible by 240)
    if len(updated_data) < total_items:
        for i in range(len(updated_data), total_items):
            updated_data.append(idioms_data[i])
    
    # Create output directory if it doesn't exist
    output_dir = "updated_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the updated data to a new JSON file
    with open(os.path.join(output_dir, 'fixed_idioms_data.json'), 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    
    # Also write a report of all corrections made
    with open(os.path.join(output_dir, 'correction_report.json'), 'w', encoding='utf-8') as f:
        json.dump(corrections_applied, f, ensure_ascii=False, indent=2)
    
    print(f"Data fix completed.")
    print(f"Fixed {indices_fixed} items with missing indices.")
    print(f"Fixed {expressions_fixed} items with incorrect expressions.")
    print(f"Updated data saved to {output_dir}/fixed_idioms_data.json")
    print(f"Correction report saved to {output_dir}/correction_report.json")

if __name__ == "__main__":
    fix_data()
