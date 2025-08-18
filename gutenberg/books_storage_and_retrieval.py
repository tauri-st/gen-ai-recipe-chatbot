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