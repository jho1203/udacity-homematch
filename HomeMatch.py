#A starter file for the HomeMatch application if you want to build your solution in a Python program instead of a notebook. 

import os
import json
from generate_listings import generate_listings
from vector_database import query_similar_listings, setup_vector_database_from_listings
from personalized_descriptions import generate_personalized_listings
from metadata_extraction import extract_search_parameters_llm

# Set environment variables for OpenAI API
os.environ["OPENAI_API_KEY"] = "voc-179973988312667737828436792a9844e21d5.28199995"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

def load_or_generate_listings(model_name="gpt-4o", temperature=0.0, max_tokens=1000):
    """Load existing listings from file or generate new ones if file doesn't exist
    
    Args:
        model_name: Name of the OpenAI model to use (default: "gpt-4o")
        temperature: Temperature parameter for the LLM (default: 0.0)
        max_tokens: Maximum number of tokens for the LLM response (default: 1000)
        
    Returns:
        list: The loaded or generated listings
    """
    # Check if listings already exist
    listings_file = 'berlin_real_estate_listings.json'
    if os.path.exists(listings_file):
        print(f"Found existing listings in {listings_file}")
        
        # Load existing listings
        with open(listings_file, 'r') as f:
            listings = json.load(f)
        print(f"Loaded {len(listings)} existing listings")
    else:
        print("No existing listings found. Generating new listings...")
        
        # Call the listing generation function from generate_listings.py with parameters
        listings = generate_listings(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    return listings

def setup_vector_database(listings=None):
    """Set up the vector database from listings
    
    Args:
        listings: List of listing texts (default: None)
        
    Returns:
        Chroma: The Chroma vector store
    """
    print("\nSetting up vector database...")
    # Use the function from vector_database.py with force_rebuild=True
    vectorstore = setup_vector_database_from_listings(force_rebuild=True)
    print("Vector database setup complete!")
    return vectorstore

def find_matching_listings(vectorstore, user_preferences, n_results=3):
    """Find listings that match the user's preferences
    
    Args:
        vectorstore: The Chroma vector store
        user_preferences: String describing the user's preferences
        n_results: Number of results to return (default: 3)
        
    Returns:
        list: List of (Document, score) tuples
    """
    print(f"\nSearching for listings matching user preferences...")
    
    # Extract metadata filters from user preferences using LLM
    metadata_filters = extract_search_parameters_llm(user_preferences)
    
    # Print the extracted filters
    if metadata_filters:
        print("Applying metadata filters:")
        for key, value in metadata_filters.items():
            print(f"  - {key}: {value}")
        
        # Try with metadata filters first
        results = query_similar_listings(
            vectorstore, 
            user_preferences, 
            n_results=n_results,
            metadata_filters=metadata_filters,
            preference_weights=None
        )
        
        # If no results with filters, fall back to semantic search
        if not results:
            print("No matches found with metadata filters, falling back to semantic search...")
            results = query_similar_listings(
                vectorstore, 
                user_preferences, 
                n_results=n_results,
                metadata_filters=None,
                preference_weights=None
            )
    else:
        # No metadata filters, just do semantic search
        results = query_similar_listings(
            vectorstore, 
            user_preferences, 
            n_results=n_results,
            metadata_filters=None,
            preference_weights=None
        )
    
    print(f"Found {len(results)} matching listings")
    return results

def collect_user_preferences():
    """Collect user preferences through a series of questions
    
    Returns:
        tuple: (questions, answers, combined_preferences)
    """
    # Define the questions to ask the user
    questions = [
        "How big do you want your apartment to be?",
        "What are 3 most important things for you in choosing this property?",
        "Which amenities would you like?",
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?"
    ]
    
    # Define the answers (hardcoded for now)
    answers = [
        "A modern two-bedroom apartment with a spacious living room and a balcony.",
        "A trendy neighborhood, good nightlife, and proximity to other young professionals.",
        "A fully equipped kitchen, high-speed internet, and a gym in the building.",
        "Close to U-Bahn and S-Bahn stations, bike lanes, and car sharing options.",
        "Very urban with lots of restaurants, bars, cafes, and cultural venues within walking distance."
    ]
    
    # Display the Q&A to the user
    print("\nUser Preference Profile:")
    print("-" * 30)
    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {answer}")
        print()
    
    # Combine the answers into a single preference string
    combined_preferences = "\n".join([f"{q}: {a}" for q, a in zip(questions, answers)])
    
    return questions, answers, combined_preferences

def main():
    """Main function to run the HomeMatch application"""
    # Print welcome message
    print("Welcome to HomeMatch - Your Personalized Real Estate Listing Generator")
    print("----------------------------------------------------------------------")
    
    # Step 1: Load listings
    listings = load_or_generate_listings()
    
    # Step 2: Set up vector database
    vectorstore = setup_vector_database(listings)
    
    # Step 3: Collect user preferences
    questions, answers, combined_preferences = collect_user_preferences()
    
    # Print user preferences
    print("\nUser Preference Profile:")
    print("------------------------------")
    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {answer}")
    print("\n")
    
    # Step 4: Find matching listings
    matching_listings = find_matching_listings(vectorstore, combined_preferences)
    
    # Step 5: Generate personalized descriptions
    print("\nGenerating personalized descriptions based on user preferences...")
    personalized_listings = generate_personalized_listings(matching_listings, combined_preferences)
    
    # Step 6: Display personalized listings
    print("\n" + "=" * 50)
    print("USER PREFERENCES:")
    print("I'm looking for a property with the following characteristics:")
    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"{question}: {answer}")
    print("=" * 50)
    
    # Display each personalized listing
    for i, listing in enumerate(personalized_listings):
        print(f"\nPERSONALIZED LISTING {i+1} (Similarity: {1-listing['similarity_score']:.2f})")
        print("-" * 40)
        print(listing['personalized_description'])
        print("-" * 40)

if __name__ == "__main__":
    main()
