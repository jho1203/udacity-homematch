import os
import chromadb
import json

# Path to the ChromaDB directory
chroma_dir = "./chroma_db"

# Check if the directory exists
if not os.path.exists(chroma_dir):
    print(f"ChromaDB directory '{chroma_dir}' does not exist.")
    exit(1)

# Initialize the ChromaDB client
client = chromadb.PersistentClient(path=chroma_dir)

# Get all collections
collections = client.list_collections()
print(f"Found {len(collections)} collections in ChromaDB.")

# Examine each collection
for collection_info in collections:
    collection_name = collection_info.name
    print(f"\nExamining collection: {collection_name}")
    
    # Get the collection
    collection = client.get_collection(name=collection_name)
    
    # Get all items in the collection
    items = collection.get()
    print(f"Collection contains {len(items['ids'])} items.")
    
    # Print some example metadata
    print("\nExample metadata entries:")
    for i, metadata in enumerate(items['metadatas']):
        if i < 5:  # Show only the first 5 items
            print(f"Item {i+1}: {json.dumps(metadata, indent=2)}")
        
    # Count items with bedroom metadata
    bedroom_count = sum(1 for metadata in items['metadatas'] if 'bedrooms' in metadata)
    print(f"\nItems with 'bedrooms' metadata: {bedroom_count} out of {len(items['metadatas'])}")
    
    # Count items with bathroom metadata
    bathroom_count = sum(1 for metadata in items['metadatas'] if 'bathrooms' in metadata)
    print(f"Items with 'bathrooms' metadata: {bathroom_count} out of {len(items['metadatas'])}")
    
    # Check the data types of bedroom and bathroom values
    if bedroom_count > 0:
        bedroom_types = set(type(metadata['bedrooms']).__name__ for metadata in items['metadatas'] if 'bedrooms' in metadata)
        print(f"Data types for 'bedrooms' field: {bedroom_types}")
    
    if bathroom_count > 0:
        bathroom_types = set(type(metadata['bathrooms']).__name__ for metadata in items['metadatas'] if 'bathrooms' in metadata)
        print(f"Data types for 'bathrooms' field: {bathroom_types}")
