import csv

def remove_duplicates_by_id(input_file, output_file=None):
    """
    Remove duplicate rows from CSV file based on the 'id' column.
    Keeps the first occurrence of each unique ID.
    """
    if output_file is None:
        output_file = input_file
    
    seen_ids = set()
    unique_rows = []
    total_rows = 0
    
    # Read the CSV file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Get the fieldnames from the reader
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_rows += 1
            row_id = row['id']
            
            # Only add row if we haven't seen this ID before
            if row_id not in seen_ids:
                seen_ids.add(row_id)
                unique_rows.append(row)
    
    # Write the unique rows back to the file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    duplicates_removed = total_rows - len(unique_rows)
    print(f"Processing complete!")
    print(f"Total rows read: {total_rows}")
    print(f"Unique rows (by ID): {len(unique_rows)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    input_file = "deputados_56_legislatura_atributos.csv"
    remove_duplicates_by_id(input_file)

