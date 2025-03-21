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
    # Set environment variables for OpenAI API
    os.environ["OPENAI_API_KEY"] = "voc-179973988312667737828436792a9844e21d5.28199995"
    os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"
    
    # Test with a sample document and preferences
    from langchain.schema import Document
    
    # Create a sample document
    sample_doc = Document(
        page_content="This beautiful apartment features high ceilings, hardwood floors, and large windows. Located in a historic building with modern amenities.",
        metadata={
            "borough": "Kreuzberg",
            "price": "€450,000",
            "bedrooms": "2",
            "bathrooms": "1",
            "size": "85 m²"
        }
    )
    
    # Sample user preferences
    sample_preferences = "I'm looking for a bright apartment with character in a lively neighborhood. I work from home so I need good natural light."
    
    # Generate personalized description
    personalized_description = create_personalized_description(sample_doc, sample_preferences)
    
    print("Sample Document:")
    print(f"Metadata: {sample_doc.metadata}")
    print(f"Content: {sample_doc.page_content}")
    
    print("\nUser Preferences:")
    print(sample_preferences)
    
    print("\nPersonalized Description:")
    print(personalized_description)
