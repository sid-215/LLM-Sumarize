 
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import langchain  # type: ignore
import openai  # type: ignore
from langchain_community.document_loaders import CSVLoader  # type: ignore
from langchain.chains import LLMChain  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain_community.llms import OpenAI  # type: ignore
from langchain_community.chat_models import ChatOpenAI  # type: ignore

from datetime import datetime


def get_summary(topic_of_interest, timeframe) :
    # Load the CSV file
    data = pd.read_csv("hmtvdata - output1.csv")

    
    
    final_data = data[(data['Tags'].str.contains(topic_of_interest, case=False, na=False)) ].sort_values(by = 'Published Time', ascending = False)
    final_data['Published Time'] = pd.to_datetime(final_data['Published Time'])
    current_time = pd.to_datetime(datetime.now()).tz_localize('Asia/Kolkata')
    final_data = final_data[final_data['Published Time'] > (current_time - pd.DateOffset(weeks=timeframe))]
    final_data = final_data[['URL','Content','Tags','Published Time']]


    
    final_data['Combined'] = "URL: " + final_data['URL'] + "\nContent: " + final_data['Content'] + "\n"

    
    with open("key2.txt", "r") as file:
        key = file.read().strip()

    


    
 
    llm = ChatOpenAI(model='gpt-4o', temperature=0, openai_api_key=key)

    
    prompt_template = """ You are a data analyst with expertise in analyzing and summarizing data. You will be given a list as an input.
    Your task is to analyze the data and provide a summary of the data in a list format in max 1-2 lines for each distinct issue along with the sources associated with each point.
    


    When a question is posed,  find the relevant information. Distinguish distinct issues in content. Do not make up information. 
    Associated URLs can be found. Add URLs as the source for each of your point if possible.
    Also, mention if any politician or political party is involved in the issue or said something about the issue if there are any.
    If you recognize it is a single issue only, do not split the issue. Give the answer in 1-2 lines.
    Search for any issues that are causing distress to people in the content.

    You need to think before answering. If you do not have any information, say "No information available" but do not make up information.
    The answer you give is crucial as politicians shaping the future of the country will depend on it and anything wrong could derail the society.

    {input_text}

    Question : What are the issues related to {topic_of_interest}?

    
    """

    
    prompt = PromptTemplate(template=prompt_template, input_variables=["input_text","topic_of_interest" ])
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(input_text=final_data['Combined'].tolist(), topic_of_interest = topic_of_interest, model='gpt-4o')



 



