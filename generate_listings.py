import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Ensure environment variables are set
os.environ["OPENAI_API_KEY"] = "voc-179973988312667737828436792a9844e21d5.28199995"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

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
    """Generate real estate listings for Berlin properties
    
    Args:
        num_listings: Number of listings to generate (default: 20)
        output_file: JSON file to save the listings to (default: 'berlin_real_estate_listings.json')
        model_name: Name of the OpenAI model to use (default: "gpt-4o")
        temperature: Temperature parameter for the LLM (default: 0.0)
        max_tokens: Maximum number of tokens for the LLM response (default: 1000)
        
    Returns:
        list: The generated listings
    """
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

# This block only runs when the script is executed directly, not when imported
if __name__ == "__main__":
    listings = generate_listings()
    
    # Display the first listing as an example
    print("\nExample listing:")
    print(listings[0])
