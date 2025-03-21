import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def create_personalized_description(listing_doc, user_preferences, model_name="gpt-4o", temperature=0.0, max_tokens=1000):
    # Initialize the LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Extract metadata and content from the listing document
    metadata = listing_doc.metadata
    content = listing_doc.page_content
    
    # Create a prompt template for generating personalized descriptions
    personalization_template = """
    You are a real estate agent tasked with creating a personalized property description for a potential buyer.
    
    Original Property Listing:
    Borough: {borough}
    Price: {price}
    Bedrooms: {bedrooms}
    Bathrooms: {bathrooms}
    Size: {size}
    
    Original Description:
    {description}
    
    Buyer's Preferences:
    {preferences}
    
    Your task is to rewrite the property description to highlight aspects that would appeal to this specific buyer based on their preferences.
    The personalized description should:
    1. Be approximately the same length as the original
    2. Emphasize features that match the buyer's preferences
    3. Maintain a professional, enthusiastic tone
    4. Include all the basic property information (borough, price, bedrooms, bathrooms, size)
    5. Be factual and only include information from the original listing
    
    Personalized Description:
    """
    
    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["borough", "price", "bedrooms", "bathrooms", "size", "description", "preferences"],
        template=personalization_template
    )
    
    # Format the prompt with the listing information and user preferences
    formatted_prompt = prompt.format(
        borough=metadata.get("borough", ""),
        price=metadata.get("price", ""),
        bedrooms=metadata.get("bedrooms", ""),
        bathrooms=metadata.get("bathrooms", ""),
        size=metadata.get("size", ""),
        description=content,
        preferences=user_preferences
    )
    
    # Generate the personalized description
    personalized_description = llm.invoke(formatted_prompt).content
    
    return personalized_description.strip()

def generate_personalized_listings(matched_listings, user_preferences, model_name="gpt-4o", temperature=0.0, max_tokens=1000):
    # Create a list to store personalized descriptions
    personalized_listings = []
    
    for i, (doc, score) in enumerate(matched_listings):
        print(f"Generating personalized description for listing {i+1}/{len(matched_listings)}...")
        
        # Create personalized description
        personalized_description = create_personalized_description(
            listing_doc=doc,
            user_preferences=user_preferences,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Add to list with original document and score
        personalized_listings.append({
            "original_doc": doc,
            "similarity_score": score,
            "personalized_description": personalized_description
        })
    
    return personalized_listings

# This block only runs when the script is executed directly, not when imported
if __name__ == "__main__":
    # Check if environment variables are set
    if "OPENAI_API_KEY" not in os.environ or "OPENAI_API_BASE" not in os.environ:
        print("Warning: OPENAI_API_KEY or OPENAI_API_BASE environment variables are not set.")
        print("Please set these environment variables before running the application.")
    
    # Import necessary modules for testing
    from langchain.schema import Document
    
    # Create a sample listing document
    listing_doc = Document(
        page_content="Borough: Kreuzberg\nPrice: €450,000\nBedrooms: 2\nBathrooms: 1\nSize: 85 m²\n\nDescription: Welcome to this stylish Altbau apartment in the heart of vibrant Kreuzberg. This beautifully renovated 2-bedroom, 1-bathroom home features high ceilings, original hardwood floors, and large windows that flood the space with natural light. The modern kitchen is equipped with high-end appliances and opens to a cozy balcony overlooking a quiet courtyard. Original architectural details have been carefully preserved while modern amenities ensure comfortable city living. The apartment includes a cellar storage space and is located in a well-maintained historic building with a newly renovated façade.\n\nNeighborhood Description: Kreuzberg is one of Berlin's most diverse and culturally rich boroughs, known for its alternative scene, vibrant nightlife, and multicultural atmosphere. The apartment is steps away from the picturesque Landwehr Canal, perfect for summer picnics and leisurely walks. Enjoy the famous Turkish Market, countless international restaurants, trendy cafés, and independent boutiques. With excellent public transportation connections via the nearby Görlitzer Bahnhof U-Bahn station, you can easily reach all parts of Berlin.",
        metadata={
            "borough": "Kreuzberg",
            "price": 450000,
            "bedrooms": "2",
            "bathrooms": "1",
            "size": 85
        }
    )
    
    # Create sample user preferences
    user_preferences = "I'm looking for a modern apartment in a trendy neighborhood with good nightlife. I need 2 bedrooms and would like to be close to public transportation. I love having outdoor space and natural light."
    
    # Generate a personalized description
    personalized_description = create_personalized_description(listing_doc, user_preferences)
    
    # Print the results
    print("\nOriginal Listing:")
    print(listing_doc.page_content[:200] + "...")
    print("\nUser Preferences:")
    print(user_preferences)
    print("\nPersonalized Description:")
    print(personalized_description)
