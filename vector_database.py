import os
import json
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

def extract_listing_metadata(listing_text):
    metadata = {}
    lines = listing_text.strip().split('\n')
    
    # Process the first section (basic information)
    for i, line in enumerate(lines):
        if ':' in line and i < 5:  # First 5 lines should contain the basic info
            key, value = line.split(':', 1)
            key = key.strip().lower()  # Ensure keys are lowercase
            value = value.strip()
            
            # Process specific metadata fields for better filtering
            if key == 'bedrooms':
                # Extract just the number for bedrooms
                bedroom_match = re.search(r'(\d+)', value)
                if bedroom_match:
                    metadata[key] = int(bedroom_match.group(1))  # Store as integer
                else:
                    metadata[key] = value
            elif key == 'bathrooms':
                # Extract just the number for bathrooms
                bathroom_match = re.search(r'(\d+)', value)
                if bathroom_match:
                    metadata[key] = int(bathroom_match.group(1))  # Store as integer
                else:
                    metadata[key] = value
            elif key == 'size':
                # Extract just the number for size
                size_match = re.search(r'(\d+)', value)
                if size_match:
                    metadata[key] = int(size_match.group(1))  # Store as integer
                else:
                    metadata[key] = value
            elif key == 'price':
                # Extract just the number for price
                price_match = re.search(r'€([\d,]+)', value)
                if price_match:
                    # Remove commas and convert to integer
                    price_str = price_match.group(1).replace(',', '')
                    metadata[key] = int(price_str)
                else:
                    metadata[key] = value
            else:
                metadata[key] = value
    
    return metadata

def setup_vector_database_from_listings(listings=None):
    db_path = "./chroma_db"
    
    # Check if database exists
    db_exists = os.path.exists(db_path) and os.path.isdir(db_path) and len(os.listdir(db_path)) > 0
    
    if not db_exists:
        # Check if listings are provided and non-empty
        if listings is None or len(listings) == 0:
            raise ValueError("Listings parameter must be provided and non-empty")
        print("Building vector database...")
        # Create embeddings for the listings
        documents = []
        metadatas = []
        ids = []
        
        for i, listing in enumerate(listings):
            # If the listing is already a string, use it directly
            # Otherwise, try to extract the listing_text field
            if isinstance(listing, dict):
                listing_text = listing.get('listing_text', '')
            else:
                listing_text = listing
            
            # Extract metadata from the listing
            metadata = extract_listing_metadata(listing_text)
            
            # Add the listing to the documents
            documents.append(listing_text)
            metadatas.append(metadata)
            ids.append(f"listing_{i}")
        
        # Initialize the embedding function
        embedding_function = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE")
        )
        
        # Create and persist the vector database
        vectorstore = Chroma.from_texts(
            documents,
            embedding_function,
            metadatas=metadatas,
            ids=ids,
            persist_directory=db_path
        )
        
        # Persist the database
        vectorstore.persist()
        
        print(f"Added {len(documents)} listings to vector database")
        return vectorstore
    
    # Try to load existing database
    try:
        print(f"Loading existing vector database from {db_path}")
        embedding_function = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE")
        )
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        print(f"Loaded {vectorstore._collection.count()} documents from vector database")
        return vectorstore
    except Exception as e:
        print(f"Error loading existing database: {e}")
        print("Rebuilding vector database...")

def query_similar_listings(vectorstore, query_text, n_results=3, metadata_filters=None):
    # Apply metadata filters if provided
    if metadata_filters:
        filter_dict = {}
        for key, value in metadata_filters.items():
            if key == "bedrooms":
                # For bedrooms, use greater than or equal to
                try:
                    # Keep as string for comparison since it's stored as string in ChromaDB
                    filter_dict[key] = value
                except ValueError:
                    # If not a valid value, skip this filter
                    print(f"  - skipping invalid {key} value: {value}")
        
        # Perform the search with filters
        try:
            results = vectorstore.similarity_search_with_score(
                query_text,
                k=n_results,  # Get more results initially
                filter=filter_dict
            )
            print(f"Found {len(results)} results with metadata filters")
        except Exception as e:
            print(f"Error applying metadata filters: {e}")
            print("Falling back to semantic search without filters")
            results = vectorstore.similarity_search_with_score(
                query_text,
                k=n_results  # Get more results initially
            )
    else:
        # No metadata filters, just do semantic search
        results = vectorstore.similarity_search_with_score(
            query_text,
            k=n_results  # Get more results initially
        )
    
    return results[:n_results]

# This block only runs when the script is executed directly, not when imported
if __name__ == "__main__":
    # Check if environment variables are set
    if "OPENAI_API_KEY" not in os.environ or "OPENAI_API_BASE" not in os.environ:
        print("Error: OPENAI_API_KEY and OPENAI_API_BASE environment variables must be set.")
        exit(1)
    
    # Load listings from file
    with open("berlin_real_estate_listings.json", "r") as f:
        listings = json.load(f)
    
    # Set up the vector database
    vectorstore = setup_vector_database_from_listings(listings)
    
    # Test a query
    test_query = "Modern apartment in a trendy neighborhood with good nightlife"
    results = query_similar_listings(vectorstore, test_query, metadata_filters={"bedrooms": "2"})
    
    print(f"\nTest query: {test_query}")
    print(f"Found {len(results)} similar listings:")
    
    for i, (doc, score) in enumerate(results):
        print(f"\nListing {i+1} (Similarity: {score:.2f})")
        print(f"Borough: {doc.metadata.get('borough', 'N/A')}")
        print(f"Bedrooms: {doc.metadata.get('bedrooms', 'N/A')}")
        print(f"Bathrooms: {doc.metadata.get('bathrooms', 'N/A')}")
        print(f"Price: {doc.metadata.get('price', 'N/A')}")
        print(f"Size: {doc.metadata.get('size', 'N/A')}")
        
        print("\nDescription:")
        # Print just the first 150 characters of the description
        print(doc.page_content[:150] + "...")
