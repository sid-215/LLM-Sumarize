
from supabase import create_client
import google.generativeai as genai
import os
import json
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import List
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env


class ArticleOutput(BaseModel):
    ID: int
    Topics: List[str]
    People: List[str]
    Places: List[str]
    Translation: str


# Supabase Credentials (Replace with your actual Supabase URL and Key)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")



# Create Supabase Client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


# Function to Get All User-Created Tables
# It is just to check the supabase connection. If this returns error, the access to database can be debugged.
def get_tables():
    try:
        response = supabase.table("hmtv_data").select("*").limit(1).execute()
        if response.data:
            return ["original_articles"]  # Replace with actual table names
        return []
    except Exception as e:
        print(f"Error fetching tables: {e}")
        return []


# Function to Fetch Articles in Batches
# Reads ID and Content   
def fetch_article_contents(batch_number=0, batch_size=10):
    try:
        offset = batch_number * batch_size  # Calculate the offset
        response = supabase.table("hmtv_data") \
            .select("ID", "Content") \
            .order("ID") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []


# Function to Process Articles with a Separator Between Each Article for LLM to process each article independently.
# It also adds ID to each article for identification.
# This is important as the LLM will process each article independently and we need to keep track of which article is which.
# The separator is used to distinguish between different articles in the input text.
def format_articles_with_separator(articles):
    try:
        # Use a compact separator between each article's content
        separator = "\n###END###\n"
        
        # Format each article to include both ID and Content
        formatted_articles = []
        for article in articles:
            article_text = f"ID: {article['ID']}\n{article['Content']}"
            formatted_articles.append(article_text)
        
        # Join all formatted articles with the separator
        formatted_text = separator.join(formatted_articles)
        return formatted_text
    except Exception as e:
        print(f"Error formatting articles: {e}")
        return ""
    
## Uses Pydantic to validate the response from the LLM.
## Making sure that the response is in the expected format and contains all the required fields.
## This is important as the response will be used to update the database.
def validate_cleaned_response(cleaned_response: str):
    try:
        # First, parse the cleaned JSON string
        parsed_data = json.loads(cleaned_response)  # List of dicts
        
        validated_articles = []

        for idx, article in enumerate(parsed_data):
            try:
                # Validate each article using Pydantic
                validated = ArticleOutput(**article)
                validated_articles.append(validated.dict())  # Convert model back to simple dict
            except ValidationError as ve:
                print(f"Validation error at index {idx}: {ve}")

        return validated_articles

    except json.JSONDecodeError as je:
        print(f"JSON decoding failed: {je}")
        return []


# This the core function that processes the articles using the Gemini 2.0 Flash API.
# Exceptions are used to log where exactly are we facing issues.
def process_with_custom_prompt(articles, custom_prompt):
    try:
        formatted_text = format_articles_with_separator(articles)
        prompt = custom_prompt.format(articles=formatted_text)
        response = model.generate_content(prompt)

        cleaned_response = response.text.replace('```json', '').replace('```', '').strip()

        if not cleaned_response:
            print("Received an empty or invalid response.")
            return []

        
        validated_data = validate_cleaned_response(cleaned_response)

        if not validated_data:
            print("Validation failed: No valid articles found.")
            return []

        return validated_data  # Safe validated articles ready for DataFrame or further processing

    except Exception as e:
        print(f"Unexpected error: {e}")
        return []





# Main Execution
if __name__ == "__main__":
    processed_df = pd.DataFrame()

    batch_size = 10
    total_batches = 50  # Load batch_size * total_batches articles (batch_size at a time)

    for batch_number in range(total_batches):
        print(f"Fetching batch {batch_number + 1} of {total_batches}...")
        articles = fetch_article_contents(batch_number, batch_size)

        if articles:
            # Example custom prompt (you can pass your own)
            custom_prompt = """
            The following are multiple articles separated by '###END###'
            For each article, I need you to generate a structured json output.
            Given the following articles extract the most relevant topics. 
            Focus on high-level topics that capture the primary narrative of the article
            Try keeping the topics concise and relevant to the article's content.
            Keep the topic within 2 words but you can expand only if you think it is necessary.
            Topics should also contain any entities, organizations mentioned.
            The topics should be a list of keywords or phrases that summarize the main themes or subjects of the article.
            Topics may also include events occurred in the article, but should not be limited to them. 
            The topics should primarily describe the core issues or themes discussed in the article.
            Do not include information about legal procedure, administrative details or any routine or general details.
            Identify the main themes and entities involved, excluding minor or tangential events.
            Also identify all the people and all the places referred to in each article. 
            The output for Topics, Places, People should be in English language only.
            Transalate the article into English and include it in the output at Translation key.
            I am also providing the numerical ID with each article.
            Do not give backticks in output as I need to parse your output into json.
            Do not explain your reasoning. Just give the output
            Articles:
            {articles}
            Output should be a json array with each element containing the following keys:
            - 'ID': only numeric id of the article that has been provided
            - 'Topics': a list of topics
            - 'People': a list of people    
            - 'Places': a list of places
            - 'Translation' : english translation of the article
            """
            processed_data = process_with_custom_prompt(articles, custom_prompt)


            if processed_data:
                    # Assuming processed_data is a dict or list of dicts
                    batch_df = pd.DataFrame(processed_data)  # Convert processed data to DataFrame
                    processed_df = pd.concat([processed_df, batch_df], ignore_index=True)
                    

 
    
    
df = processed_df
    
df['Topics'] = df['Topics'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x).str.replace('[', '').str.replace(']', '')
df['People'] = df['People'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x).str.replace('[', '').str.replace(']', '')
df['Places'] = df['Places'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x).str.replace('[', '').str.replace(']', '')


id_list = df['ID'].tolist()

for index, row in df.iterrows():
    id_value = row['ID']
    
    # Create the update data
    update_data = {
        'Topics': row['Topics'],
        'People': row['People'],
        'Places': row['Places'],
        'Translation': row['Translation']
    }
    
    # Update the Supabase record
    supabase.table('hmtv_data') \
        .update(update_data) \
        .eq('ID', id_value) \
        .execute()



            

"""response = supabase.table('hmtv_data') \
    .select('*') \
    .in_('ID', id_list) \
    .execute()
supabase_df = pd.DataFrame(response.data)"""


"""id_list = df['ID'].tolist()

for index, row in df.iterrows():
    id_value = row['ID']
    
    # Create the update data
    update_data = {
        'Topics': row['Topics'],
        'People': row['People'],
        'Places': row['Places'],
        'Translation': row['Translation']
    }
    
    # Update the Supabase record
    supabase.table('hmtv_data') \
        .update(update_data) \
        .eq('ID', id_value) \
        .execute()"""




