import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

def extract_search_parameters_llm(user_preferences, model_name="gpt-4o", temperature=0.0):
    # Create a parser for the output
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="bedrooms", description="The number of bedrooms required (as a string). Only include if explicitly mentioned.", required=False),
        ResponseSchema(name="bathrooms", description="The number of bathrooms required (as a string). Only include if explicitly mentioned.", required=False)
    ])
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that extracts search parameters from user preferences."),
        HumanMessagePromptTemplate.from_template(
            "Extract the following information from the user preferences:\n\n"
            "User Preferences: {user_preferences}\n\n"
            "Extract bedrooms and bathrooms if explicitly mentioned. "
            "If not mentioned, include the field with an empty string value. "
            "Always include both fields in your response, even if empty.\n\n"
            "Important: Return valid JSON without any comments or trailing commas.\n\n"
            "{format_instructions}"
        )
    ])
    
    # Format the prompt with the user preferences
    format_instructions = parser.get_format_instructions()
    formatted_prompt = prompt.format_messages(
        user_preferences=user_preferences,
        format_instructions=format_instructions
    )
    
    # Create a chat model
    chat_model = ChatOpenAI(model=model_name, temperature=temperature)
    
    # Get the response
    response = chat_model.invoke(formatted_prompt)
    print(f"LLM response: {response.content}")
    
    # Parse the response
    try:
        metadata_filters = parser.parse(response.content)
        # Remove empty values
        metadata_filters = {k: v for k, v in metadata_filters.items() if v}
        return metadata_filters
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response content: {response.content}")
        return {}


# Test function for isolated testing
def test_extraction():
    # Test cases
    test_preferences = [
        "I want a modern two-bedroom apartment with at least one bathroom in Mitte or Kreuzberg.",
        "Looking for a spacious 3-bedroom place with 2 bathrooms and at least 100 square meters.",
        "I need an apartment close to public transportation with a balcony.",
        "I'm looking for a place with 2 bedrooms.",  # Test case with only bedrooms
        "I need a home with at least 1 bathrooms."  # Test case with only bathrooms
    ]
    
    for i, pref in enumerate(test_preferences, 1):
        print(f"Test {i}: {pref}")
        filters = extract_search_parameters_llm(pref)
        print(f"Extracted filters: {filters}\n")


# Allow running this file directly for testing
if __name__ == "__main__":
    # Check if environment variables are set
    if "OPENAI_API_KEY" not in os.environ or "OPENAI_API_BASE" not in os.environ:
        print("Warning: OPENAI_API_KEY or OPENAI_API_BASE environment variables are not set.")
        print("Please set these environment variables before running the application.")
    
    # Run the test function
    test_extraction()
