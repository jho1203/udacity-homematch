import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Define a prompt template for generating real estate listings in Berlin, Germany
listing_template = """
Generate a detailed real estate listing for a property in Berlin, Germany with the following components:
1. Basic information (borough, price, bedrooms, bathrooms, apartment size in square meters)
2. A detailed property description highlighting unique features
3. A neighborhood description specific to that Berlin borough

The listing should be for a {property_type} in the {borough} borough of Berlin with {bedrooms} bedrooms.

Use the metric system (square meters, not square feet) and Euros (€) for the price. Match price and apartment sizes to what's more or less common for each borough (based on average income in those areas)

Format the output exactly like this example, but with different content specific to Berlin:

Borough: Kreuzberg
Price: €450,000
Bedrooms: {bedrooms}
Bathrooms: 1
Size: 85 m²

Description: Welcome to this stylish Altbau apartment in the heart of vibrant Kreuzberg. This beautifully renovated {bedrooms}-bedroom, 1-bathroom home features high ceilings, original hardwood floors, and large windows that flood the space with natural light. The modern kitchen is equipped with high-end appliances and opens to a cozy balcony overlooking a quiet courtyard. Original architectural details have been carefully preserved while modern amenities ensure comfortable city living. The apartment includes a cellar storage space and is located in a well-maintained historic building with a newly renovated façade.

Neighborhood Description: Kreuzberg is one of Berlin's most diverse and culturally rich boroughs, known for its alternative scene, vibrant nightlife, and multicultural atmosphere. The apartment is steps away from the picturesque Landwehr Canal, perfect for summer picnics and leisurely walks. Enjoy the famous Turkish Market, countless international restaurants, trendy cafés, and independent boutiques. With excellent public transportation connections via the nearby Görlitzer Bahnhof U-Bahn station, you can easily reach all parts of Berlin.
"""

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["property_type", "borough", "bedrooms"],
    template=listing_template
)

# Define different property types common in Berlin
property_types = [
    "Old building (Altbau) apartment", 
    "modern penthouse", 
    "garden apartment", 
    "converted loft",
    "luxury Neubau apartment",
    "historic Berliner Zimmer flat",
    "canal-side apartment",
    "artist studio apartment",
    "east-german (Plattenbau) apartment"
]

# Define Berlin's distinct boroughs
berlin_boroughs = [
    "Mitte", 
    "Kreuzberg", 
    "Prenzlauer Berg", 
    "Charlottenburg", 
    "Neukölln",
    "Friedrichshain",
    "Schöneberg",
    "Wedding",
    "Moabit",
    "Wilmersdorf"
]

def generate_listings(num_listings=20, output_file='berlin_real_estate_listings.json', model_name="gpt-4o", temperature=0.0, max_tokens=1000):
    # Initialize the LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Generate listings
    listings = []

    bedroom_counts = [1, 2, 3, 4]
    for i in range(num_listings):
        # Select property type, borough, and bedroom count (cycling through the lists)
        property_type = property_types[i % len(property_types)]
        borough = berlin_boroughs[i % len(berlin_boroughs)]
        bedrooms = bedroom_counts[i % len(bedroom_counts)]
        
        # Format the prompt with the selected types
        formatted_prompt = prompt.format(
            property_type=property_type,
            borough=borough,
            bedrooms=bedrooms
        )
        
        # Generate the listing using the LLM
        listing = llm.invoke(formatted_prompt).content
        
        # Add to our collection
        listings.append(listing.strip())
        
        # Print progress
        print(f"Generated listing {i+1}/{num_listings}")

    # Save the listings to a JSON file for later use
    with open(output_file, 'w') as f:
        json.dump(listings, f, indent=2)

    print(f"\nAll listings generated and saved to '{output_file}'")
    
    return listings

def load_or_generate_listings(listings_file='berlin_real_estate_listings.json', num_listings=20, model_name="gpt-4o", temperature=0.0, max_tokens=1000):
    """Load existing listings from file or generate new ones if the file doesn't exist.
    
    Args:
        listings_file (str): Path to the listings JSON file
        num_listings (int): Number of listings to generate if file doesn't exist
        model_name (str): Name of the OpenAI model to use for generation
        temperature (float): Temperature parameter for generation
        max_tokens (int): Maximum tokens for generation
        
    Returns:
        list: A list of listing strings
    """
    # Check if listings already exist
    if os.path.exists(listings_file):
        print(f"Found existing listings in {listings_file}")
        
        # Load existing listings
        with open(listings_file, 'r') as f:
            listings = json.load(f)
        print(f"Loaded {len(listings)} existing listings")
    else:
        print("No existing listings found. Generating new listings...")
        
        # Call the listing generation function
        listings = generate_listings(
            num_listings=num_listings,
            output_file=listings_file,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    # Ensure consistent return type (list of strings)
    if listings and isinstance(listings[0], dict):
        # If listings are dictionaries, extract the listing_text field
        return [listing.get('listing_text', '') for listing in listings]
    
    return listings


def test_load_or_generate_listings():
    """Test function to verify that load_or_generate_listings returns a consistent type."""
    # Create a temporary test file
    test_file = 'test_listings.json'
    
    # Test case 1: No file exists yet
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # If the main listings file exists, copy it to the test file to avoid API calls
    if os.path.exists(test_file):
        print(f"Using existing listings from {test_file} for testing")
        with open(test_file, 'r') as src:
            listings_data = json.load(src)
            # Only use 2 listings for the test
            test_data = listings_data[:2] if len(listings_data) >= 2 else listings_data
            with open(test_file, 'w') as dest:
                json.dump(test_data, dest)
        
        listings0 = load_or_generate_listings(listings_file=test_file)
    else:
        # Generate just 2 listings to keep the test fast
        print("No existing listings found. Generating 2 new listings for testing...")
        listings0 = load_or_generate_listings(listings_file=test_file, num_listings=2)
    
    # Verify type
    assert isinstance(listings0, list), "Listings should be a list"
    if listings0:
        assert isinstance(listings0[0], str), "Each listing should be a string"
    
    # Test case 2: Load existing listings
    listings1 = load_or_generate_listings(listings_file=test_file)
    
    # Verify type
    assert isinstance(listings1, list), "Listings should be a list"
    if listings1:
        assert isinstance(listings1[0], str), "Each listing should be a string"
    
    # Test case 3: Load again to verify consistency
    listings2 = load_or_generate_listings(listings_file=test_file)
    
    # Verify type
    assert isinstance(listings2, list), "Loaded listings should be a list"
    if listings2:
        assert isinstance(listings2[0], str), "Each loaded listing should be a string"
    
    # Verify both have the same structure
    assert len(listings1) == len(listings2), "Both should have the same number of listings"
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("All tests passed! load_or_generate_listings returns consistent types.")


# This block only runs when the script is executed directly, not when imported
if __name__ == "__main__":
    # Check if environment variables are set
    if "OPENAI_API_KEY" not in os.environ or "OPENAI_API_BASE" not in os.environ:
        print("Warning: OPENAI_API_KEY or OPENAI_API_BASE environment variables are not set.")
        print("Please set these environment variables before running the application.")
    
    # Run the test function only
    test_load_or_generate_listings()
