import os
import json
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

def extract_listing_data(listing):
    """Extract structured data from a listing text
    
    Args:
        listing: The listing text
        
    Returns:
        Document: Document with metadata
    """
    # Import re at the top of the function to avoid multiple imports
    import re
    
    metadata = {}
    lines = listing.strip().split('\n')
    document_parts = []
    
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
                    metadata[key] = bedroom_match.group(1)  # Store as string for consistency
                else:
                    metadata[key] = value
            elif key == 'bathrooms':
                # Extract just the number for bathrooms
                bathroom_match = re.search(r'(\d+)', value)
                if bathroom_match:
                    metadata[key] = bathroom_match.group(1)  # Store as string for consistency
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
        elif line.startswith("Description:"):  # Start of description
            description = line.replace("Description:", "").strip()
            document_parts.append(description)
        elif line.startswith("Neighborhood Description:"):  # Start of neighborhood description
            neighborhood = line.replace("Neighborhood Description:", "").strip()
            document_parts.append(neighborhood)
        elif i > 5 and not line.startswith("Description:") and not line.startswith("Neighborhood Description:"):
            # This is part of a description
            document_parts.append(line)
    
    # Join all parts of the document text
    document_text = " ".join(document_parts)
    
    # Print the extracted metadata for debugging
    print(f"Extracted metadata: {metadata}")
    
    return Document(page_content=document_text, metadata=metadata)

def setup_vector_database(listings, persist_directory="./chroma_db"):
    """Set up the vector database with listings using LangChain's Chroma integration
    
    Args:
        listings: List of listing texts
        persist_directory: Directory to persist the database (default: "./chroma_db")
        
    Returns:
        Chroma: The Chroma vector store
    """
    # Create documents from listings
    documents = [extract_listing_data(listing) for listing in listings]
    
    # Initialize the OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE")
    )
    
    # Create or get the Chroma vector store
    # If the persist_directory exists, it will load the existing DB
    # Otherwise, it will create a new one with the provided documents
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print(f"Loading existing vector database from {persist_directory}")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print(f"Loaded {vectorstore._collection.count()} documents from vector database")
    else:
        print(f"Creating new vector database in {persist_directory}")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        print(f"Added {len(documents)} documents to vector database")
    
    return vectorstore

def setup_vector_database_from_listings(listings_file="berlin_real_estate_listings.json", force_rebuild=False):
    """Set up the vector database from listings file
    
    Args:
        listings_file: Path to the listings file (default: "berlin_real_estate_listings.json")
        force_rebuild: Whether to force a rebuild of the vector database (default: False)
        
    Returns:
        Chroma: The Chroma vector store
    """
    # Check if we have existing listings
    if os.path.exists(listings_file):
        print(f"Found existing listings in {listings_file}")
        with open(listings_file, "r") as f:
            listings = json.load(f)
        print(f"Loaded {len(listings)} existing listings")
        
        # Print an example original listing
        print("\nExample original listing:")
        print(listings[0])
    else:
        print(f"No existing listings found in {listings_file}")
        listings = []
    
    # Check if we should rebuild the database
    db_path = "./chroma_db"
    if force_rebuild or not os.path.exists(db_path):
        print("Building new vector database...")
        return setup_vector_database(listings, persist_directory=db_path)
    else:
        try:
            print("Loading existing vector database from ./chroma_db")
            # Initialize the embedding function
            embedding_function = OpenAIEmbeddings(
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                openai_api_base=os.environ.get("OPENAI_API_BASE")
            )
            # Load the existing database
            vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_function)
            print(f"Loaded {vectorstore._collection.count()} documents from vector database")
            return vectorstore
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print("Rebuilding vector database...")
            return setup_vector_database(listings, persist_directory=db_path)

def query_similar_listings(vectorstore, query_text, n_results=3, metadata_filters=None, preference_weights=None):
    """Query the vector store for listings similar to the query text with enhanced filtering
    
    Args:
        vectorstore: The Chroma vector store
        query_text: The query text
        n_results: Number of results to return (default: 3)
        metadata_filters: Dictionary of metadata filters to apply (e.g., {"bedrooms": "2"}) (default: None)
        preference_weights: Deprecated parameter, kept for backward compatibility
        
    Returns:
        list: The similar documents with their scores
    """
    # Extract key requirements from query text if needed
    required_features = extract_key_requirements(query_text)
    
    # Apply metadata filters if provided
    if metadata_filters:
        filter_dict = {}
        for key, value in metadata_filters.items():
            if key == "bedrooms":
                # For bedrooms, use greater than or equal to
                try:
                    # Keep as string for comparison since it's stored as string in ChromaDB
                    filter_dict[key] = value
                    print(f"  - minimum {key}: {value}")
                except ValueError:
                    # If not a valid value, skip this filter
                    print(f"  - skipping invalid {key} value: {value}")
            elif key == "bathrooms":
                # For bathrooms, use greater than or equal to
                try:
                    # Keep as string for comparison since it's stored as string in ChromaDB
                    filter_dict[key] = value
                    print(f"  - minimum {key}: {value}")
                except ValueError:
                    # If not a valid value, skip this filter
                    print(f"  - skipping invalid {key} value: {value}")
        
        # Perform the search with filters
        try:
            results = vectorstore.similarity_search_with_score(
                query_text,
                k=n_results * 3,  # Get more results initially to allow for reranking
                filter=filter_dict
            )
            print(f"Found {len(results)} results with metadata filters")
        except Exception as e:
            print(f"Error applying metadata filters: {e}")
            print("Falling back to semantic search without filters")
            results = vectorstore.similarity_search_with_score(
                query_text,
                k=n_results * 3  # Get more results initially to allow for reranking
            )
    else:
        # No metadata filters, just do semantic search
        results = vectorstore.similarity_search_with_score(
            query_text,
            k=n_results * 3  # Get more results initially to allow for reranking
        )
    
    # Rerank results based on required features
    if required_features:
        results = rerank_results(results, required_features)
    
    return results[:n_results]

def extract_key_requirements(query_text):
    """Extract key requirements from the query text
    
    Args:
        query_text: The query text
        
    Returns:
        dict: Dictionary of key requirements
    """
    required_features = {}
    
    # Look for bedroom requirements
    if "bedroom" in query_text.lower():
        # Simple pattern matching for bedroom requirements
        if "one bedroom" in query_text.lower() or "1 bedroom" in query_text.lower():
            required_features["bedrooms"] = "1"
        elif "two bedroom" in query_text.lower() or "2 bedroom" in query_text.lower():
            required_features["bedrooms"] = "2"
        elif "three bedroom" in query_text.lower() or "3 bedroom" in query_text.lower():
            required_features["bedrooms"] = "3"
    
    # Look for neighborhood preferences
    neighborhoods = ["Mitte", "Kreuzberg", "Prenzlauer Berg", "Neukölln", "Wedding", "Charlottenburg"]
    for neighborhood in neighborhoods:
        if neighborhood.lower() in query_text.lower():
            required_features["borough"] = neighborhood
    
    # Look for amenity requirements
    amenities = ["balcony", "garden", "terrace", "elevator", "parking"]
    for amenity in amenities:
        if amenity in query_text.lower():
            required_features["amenities"] = required_features.get("amenities", []) + [amenity]
    
    return required_features

def rerank_results(results, required_features=None):
    """Rerank results based on required features
    
    Args:
        results: List of (Document, score) tuples
        required_features: Dictionary of required features
        
    Returns:
        list: Reranked list of (Document, score) tuples
    """
    reranked_results = []
    
    for doc, score in results:
        # Start with the original similarity score
        adjusted_score = score
        
        # Apply penalties for missing required features
        if required_features:
            for feature, value in required_features.items():
                if feature == "bedrooms" and feature in doc.metadata:
                    # Penalize if bedrooms don't match
                    if doc.metadata[feature] != value:
                        adjusted_score += 0.1  # Increase distance (worse score)
                
                elif feature == "borough" and feature in doc.metadata:
                    # Penalize if borough doesn't match
                    if doc.metadata[feature] != value:
                        adjusted_score += 0.1  # Increase distance (worse score)
                
                elif feature == "amenities":
                    # Check if amenities are mentioned in the document content
                    for amenity in value:
                        if amenity not in doc.page_content.lower():
                            adjusted_score += 0.05  # Small penalty per missing amenity
        
        reranked_results.append((doc, adjusted_score))
    
    # Sort by adjusted score (lower is better)
    reranked_results.sort(key=lambda x: x[1])
    
    return reranked_results

# This block only runs when the script is executed directly, not when imported
if __name__ == "__main__":
    # Set environment variables for OpenAI API
    os.environ["OPENAI_API_KEY"] = "voc-179973988312667737828436792a9844e21d5.28199995"
    os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"
    
    # Load listings
    listings_file = 'berlin_real_estate_listings.json'
    with open(listings_file, 'r') as f:
        listings = json.load(f)
    
    # Set up the vector database
    vectorstore = setup_vector_database(listings)
    
    # Test a query
    test_query = "Modern apartment in a trendy neighborhood with good nightlife"
    results = query_similar_listings(vectorstore, test_query, metadata_filters={"bedrooms": "2"})
    
    print(f"\nTest query: {test_query}")
    print(f"Found {len(results)} similar listings:")
    
    for i, (doc, score) in enumerate(results):
        print(f"\nListing {i+1} (Similarity: {score:.2f})")
        print(f"Borough: {doc.metadata.get('borough', 'Unknown')}")
        print(f"Bedrooms: {doc.metadata.get('bedrooms', 'Unknown')}")
        print(f"Bathrooms: {doc.metadata.get('bathrooms', 'Unknown')}")
        print(f"Price: {doc.metadata.get('price', 'Unknown')}")
        print(f"Size: {doc.metadata.get('size', 'Unknown')}")
        print("\nDescription:")
        print(doc.page_content[:200] + "...")
