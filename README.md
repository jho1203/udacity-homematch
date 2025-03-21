# HomeMatch: Personalized Real Estate Listings Generator

HomeMatch is an innovative application that uses Large Language Models (LLMs) and vector databases to create personalized real estate listings based on buyer preferences. The system focuses on properties in Berlin, Germany, using the metric system and Euro currency.

## Features

1. **Listing Generation**: Creates detailed real estate listings for properties in Berlin, including property descriptions and neighborhood information.

2. **Vector Database Integration**: Stores property listings in a vector database (ChromaDB) for efficient similarity searching.

3. **Personalized Matching**: Finds properties that match user preferences using semantic similarity search.

4. **Personalized Descriptions**: Rewrites property descriptions to highlight aspects that match the user's preferences.

5. **Diverse Property Options**: Includes a variety of properties with 1, 2, 3, and 4 bedrooms to accommodate different user needs.

## LLM-Based Metadata Extraction

A key feature of HomeMatch is the use of an LLM to extract mandatory requirements from user preferences:

- The `extract_search_parameters_llm` function uses an LLM to analyze user preferences
- It identifies specific requirements for bedrooms and bathrooms
- These requirements are converted to metadata filters for the vector database
- The system treats numeric filters (bedrooms, bathrooms) as minimum requirements
- Metadata filtering is implemented to match string-based bedroom and bathroom values

## Recent Enhancements

- **Diverse Listings**: Generated 20 listings with an even distribution of 1, 2, 3, and 4 bedroom properties
- **Improved Metadata Handling**: Updated the metadata extraction process to properly handle bedroom and bathroom requirements
- **Optimized Storage**: Removed redundant string attributes from the database to improve efficiency
- **Better Filtering**: Fixed the metadata filtering to properly match string-based bedroom and bathroom values

## Project Structure

- `HomeMatch.py`: Main application file that orchestrates the entire process.
- `generate_listings.py`: Module for generating real estate listings using LLMs.
- `vector_database.py`: Module for setting up and querying the vector database.
- `personalized_descriptions.py`: Module for creating personalized property descriptions.
- `metadata_extraction.py`: Module for extracting structured metadata from user preferences.
- `check_chroma.py`: Utility script for inspecting and debugging the ChromaDB database.
- `berlin_real_estate_listings.json`: JSON file containing the generated listings.
- `chroma_db/`: Directory containing the vector database files.

## Debugging Tools

The project includes utilities to help with debugging and development:

### ChromaDB Inspector

The `check_chroma.py` script allows you to inspect the contents of the ChromaDB database:

```bash
python check_chroma.py
```

This tool provides information about:
- The number of collections in the database
- The number of items in each collection
- Sample metadata entries to verify proper data storage
- Statistics on metadata fields like bedrooms and bathrooms
- Data types used for each metadata field

This is particularly useful for verifying that metadata is being correctly stored and formatted for filtering operations.

## Requirements

The project requires the following dependencies:

- langchain (0.0.305)
- openai (0.28.1)
- chromadb (0.4.12)
- sentence-transformers (>=2.2.0)
- transformers (>=4.31.0)
- pydantic (>=1.10.12)

These dependencies are listed in the `requirements.txt` file and can be installed using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Set up your environment variables for the OpenAI API:
   ```
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_API_BASE="https://api.openai.com/v1"
   ```

2. Run the main application:
   ```bash
   python HomeMatch.py
   ```

3. The application will:
   - Load existing listings or generate new ones if none exist
   - Set up the vector database with the listings
   - Find properties that match the user's preferences
   - Generate personalized descriptions for the matching properties

## Customization

You can customize the user preferences by modifying the `user_preferences` variable in the `main()` function of `HomeMatch.py`. For example:

```python
user_preferences = "I'm looking for a family-friendly apartment with a garden in a quiet neighborhood. I need at least 3 bedrooms and good schools nearby."
```

To regenerate listings with different properties, you can run:

```bash
python generate_listings.py
```

This will create 20 new listings with a variety of bedroom counts (1-4) and save them to the `berlin_real_estate_listings.json` file.
