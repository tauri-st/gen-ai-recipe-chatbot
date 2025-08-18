# Import the modules
import os
import argparse
from dotenv import load_dotenv
 
# Project Gutenberg
from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import get_text_by_id
 
# LangChain & Vector Store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQAWithSourcesChain
 
# Supabase
from supabase import create_client, Client
from supabase.client import ClientOptions


#Filter the database by keywords and dates
# Constants
COOKING_KEYWORDS = ["cooking", "recipes", "cookbook", "culinary"]

def search_gutenberg_titles(cache, keywords, top_n=10, start_date=None, end_date=None):
    """
    Search Project Gutenberg for cooking-related books, optionally filtered by date.
    Returns: List of (gutenbergbookid, title).
    """
    matching_books = []

    # Variables with SQL clauses to build SQL statement
    keyword_filters = " OR ".join([f"s.name LIKE '%{kw}%'" for kw in keywords])
    
    date_filter = ""
    if start_date and end_date:
        date_filter = f"AND b.dateissued BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        date_filter = f"AND b.dateissued >= '{start_date}'"
    elif end_date:
        date_filter = f"AND b.dateissued <= '{end_date}'"
    
    # Concatenate SQL statement to search the database's books, titles, and subject tables for matching books and select book names and IDs
    query = f"""
        SELECT DISTINCT b.gutenbergbookid AS gutenbergbookid, t.name AS title
        FROM books b
        LEFT JOIN titles t ON b.id = t.bookid
        LEFT JOIN book_subjects bs ON b.id = bs.bookid
        LEFT JOIN subjects s ON bs.subjectid = s.id
        WHERE ({keyword_filters}) {date_filter}
        LIMIT {top_n};
    """

    # Run query and hold cache of results
    results = cache.native_query(query)

    # Loop through the results and append to matching_books
    for row in results:
        gutenbergbookid, title = row
        matching_books.append((gutenbergbookid, title))

    return matching_books