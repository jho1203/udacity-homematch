# HomeMatch

Find your perfect Berlin home with AI-powered personalized listings.

## What it does

HomeMatch helps you find your ideal property by understanding what you're looking for and matching it with available listings in Berlin. It uses AI to:

- Match your preferences with suitable properties
- Highlight features you care about in property descriptions
- Filter by requirements like bedrooms and bathrooms

## How to use it

1. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_API_BASE="https://openai.vocareum.com/v1"
   ```

2. Run the app:
   ```bash
   python HomeMatch.py
   ```

3. Tell it what you're looking for, and it'll show you matching properties with descriptions tailored to your preferences.

## Project files

- `HomeMatch.py`: Main application
- `generate_listings.py`: Creates property listings
- `vector_database.py`: Handles property searching
- `personalized_descriptions.py`: Customizes property descriptions
- `metadata_extraction.py`: Understands your requirements
- `check_chroma.py`: Debug tool for the database

## Requirements

- Python 3.13.0
- Dependencies in `requirements.txt` (install with `pip install -r requirements.txt`)

## Need to debug?

Run `python check_chroma.py` to see what's in the database and how properties are being stored.

## Want different listings?

Run `python generate_listings.py` to create 20 new properties with various bedroom counts.
