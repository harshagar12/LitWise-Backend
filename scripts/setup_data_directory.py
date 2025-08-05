"""
Script to set up the data directory and validate CSV files
"""

import os
import pandas as pd
from pathlib import Path

def create_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Data directory created/verified: {data_dir.absolute()}")
    return data_dir

def validate_csv_files(data_dir):
    """Validate that required CSV files exist and have correct structure"""
    required_files = {
        "books.csv": ["book_id", "goodreads_book_id", "title", "authors", "average_rating", "ratings_count"],
        "tags.csv": ["tag_id", "tag_name"],
        "book_tags.csv": ["goodreads_book_id", "tag_id", "count"],
        "ratings.csv": ["user_id", "book_id", "rating"],
        "to_read.csv": ["user_id", "book_id"]
    }
    
    validation_results = {}
    
    for filename, required_columns in required_files.items():
        filepath = data_dir / filename
        
        if not filepath.exists():
            validation_results[filename] = {
                "exists": False,
                "error": f"File not found: {filepath}"
            }
            continue
        
        try:
            # Try to read the CSV file
            df = pd.read_csv(filepath, nrows=5)  # Just read first 5 rows for validation
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_results[filename] = {
                    "exists": True,
                    "valid": False,
                    "error": f"Missing columns: {missing_columns}",
                    "found_columns": list(df.columns)
                }
            else:
                validation_results[filename] = {
                    "exists": True,
                    "valid": True,
                    "rows": len(df),
                    "columns": list(df.columns)
                }
                
        except Exception as e:
            validation_results[filename] = {
                "exists": True,
                "valid": False,
                "error": f"Error reading file: {str(e)}"
            }
    
    return validation_results

def print_validation_results(results):
    """Print validation results in a readable format"""
    print("\n" + "="*60)
    print("CSV FILE VALIDATION RESULTS")
    print("="*60)
    
    for filename, result in results.items():
        print(f"\n📁 {filename}:")
        
        if not result["exists"]:
            print(f"   ❌ {result['error']}")
        elif not result.get("valid", False):
            print(f"   ❌ {result['error']}")
            if "found_columns" in result:
                print(f"   📋 Found columns: {result['found_columns']}")
        else:
            print(f"   ✅ Valid file with {result['rows']} sample rows")
            print(f"   📋 Columns: {result['columns']}")

def create_sample_instructions():
    """Create instructions for setting up the data files"""
    instructions = """
📋 DATA SETUP INSTRUCTIONS
==========================

To use your actual Goodreads dataset, please:

1. Create a 'data' directory in your project root
2. Place your CSV files in the data directory:
   - books.csv
   - tags.csv  
   - book_tags.csv
   - ratings.csv (optional)
   - to_read.csv (optional)

3. Ensure your CSV files have the following structure:

📚 books.csv:
   - book_id, goodreads_book_id, title, authors, average_rating, ratings_count, image_url

🏷️ tags.csv:
   - tag_id, tag_name

🔗 book_tags.csv:
   - goodreads_book_id, tag_id, count

⭐ ratings.csv (optional):
   - user_id, book_id, rating

📖 to_read.csv (optional):
   - user_id, book_id

Example file structure:
project-root/
├── data/
│   ├── books.csv
│   ├── tags.csv
│   ├── book_tags.csv
│   ├── ratings.csv
│   └── to_read.csv
├── scripts/
└── app/

If files are missing or invalid, the system will automatically fall back to sample data.
"""
    
    with open("DATA_SETUP_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print(instructions)

def main():
    """Main function to set up and validate data directory"""
    print("🚀 Setting up data directory for LitWise recommendation engine...")
    
    # Create data directory
    data_dir = create_data_directory()
    
    # Validate CSV files
    print("\n🔍 Validating CSV files...")
    results = validate_csv_files(data_dir)
    
    # Print results
    print_validation_results(results)
    
    # Check if any required files are missing
    required_files = ["books.csv", "tags.csv", "book_tags.csv"]
    missing_required = [f for f in required_files if not results[f]["exists"]]
    
    if missing_required:
        print(f"\n⚠️  Missing required files: {missing_required}")
        print("The system will use sample data until you provide the actual CSV files.")
        create_sample_instructions()
    else:
        valid_required = [f for f in required_files if results[f].get("valid", False)]
        if len(valid_required) == len(required_files):
            print("\n🎉 All required CSV files are present and valid!")
            print("The recommendation engine will use your actual Goodreads data.")
        else:
            print("\n⚠️  Some CSV files have validation errors.")
            print("Please check the file formats and try again.")
            create_sample_instructions()

if __name__ == "__main__":
    main()
