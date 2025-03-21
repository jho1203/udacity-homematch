#A starter file for the HomeMatch application if you want to build your solution in a Python program instead of a notebook. 

import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from generate_listings import load_or_generate_listings
from vector_database import setup_vector_database_from_listings, query_similar_listings
from personalized_descriptions import generate_personalized_listings
from metadata_extraction import extract_search_parameters_llm

# Check if environment variables are set
if "OPENAI_API_KEY" not in os.environ or "OPENAI_API_BASE" not in os.environ:
    print("Warning: OPENAI_API_KEY or OPENAI_API_BASE environment variables are not set.")
    print("Please set these environment variables before running the application.")

def find_matching_listings(vectorstore, user_preferences, n_results=3):
    # Extract metadata filters from user preferences using LLM
    metadata_filters = extract_search_parameters_llm(user_preferences)
    
    # Print the extracted metadata filters
    if metadata_filters:
        print("Applying metadata filters:")
        for key, value in metadata_filters.items():
            print(f"  - {key}: {value}")
        
        # Try with metadata filters first
        results = query_similar_listings(
            vectorstore, 
            user_preferences, 
            n_results=n_results,
            metadata_filters=metadata_filters
        )
        
        # If no results with filters, fall back to semantic search
        if not results:
            print("No matches found with metadata filters, falling back to semantic search...")
            results = query_similar_listings(
                vectorstore, 
                user_preferences, 
                n_results=n_results,
                metadata_filters=None
            )
    else:
        # No metadata filters, just do semantic search
        results = query_similar_listings(
        vectorstore, 
        user_preferences, 
        n_results=n_results,
            metadata_filters=None
    )
    
    print(f"Found {len(results)} matching listings")
    return results

def collect_user_preferences():
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
    print("Welcome to HomeMatch - Your Personalized Real Estate Listing Generator")
    print("----------------------------------------------------------------------\n")
    
    # Step 1: Load or generate real estate listings
    print("Step 1: Loading or generating real estate listings...")
    listings = load_or_generate_listings()
    
    # Step 2: Set up vector database
    print("\nStep 2: Setting up vector database...")
    vectorstore = setup_vector_database_from_listings(listings)
    
    # Step 3: Collect user preferences
    print("\nStep 3: Collecting user preferences...")
    questions, answers, combined_preferences = collect_user_preferences()
    
    # Print the user's preference profile
    print("\nUser Preference Profile:")
    print("------------------------------")
    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {answer}")
    print("\n")
    
    # Step 4: Find matching listings
    print("\nStep 4: Finding matching listings...")
    matched_listings = find_matching_listings(vectorstore, combined_preferences)
    
    if matched_listings:
        print(f"Found {len(matched_listings)} matching listings")
        
        # Step 5: Generate personalized descriptions
        print("\nStep 5: Generating personalized descriptions...")
        personalized_listings = generate_personalized_listings(matched_listings, combined_preferences)
        
        # Step 6: Display personalized listings
        print("\nStep 6: Displaying personalized listings...")
        print("\nHere are your personalized property listings:")
        print("=============================================\n")
        
        for i, personalized_listing in enumerate(personalized_listings):
            print(f"Listing {i+1}:")
            print("----------")
            print(personalized_listing)
            print("\n")
    else:
        print("Sorry, no matching listings were found for your preferences.")
        print("Please try again with different preferences.")

if __name__ == "__main__":
    main()
