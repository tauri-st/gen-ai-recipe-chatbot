import os
import time
import json
import logging
import datetime
from dotenv import load_dotenv

# Flask imports
from flask import Flask, render_template, request, redirect, url_for, flash, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Supabase imports
from supabase import create_client
from supabase.client import ClientOptions

# LangChain imports
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.query_constructors.supabase import SupabaseVectorTranslator
from langchain.agents import tool
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List

# RAG imports
from gutenberg.books_storage_and_retrieval import (
   perform_similarity_search as perform_books_similarity_search,
   perform_retrieval_qa as perform_books_retrieval_qa,
)

from gutenberg.recipes_storage_and_retrieval_v2 import (
    perform_similarity_search as perform_recipes_similarity_search,
    perform_self_query_retrieval as perform_recipes_self_query_retrieval,
    perform_multi_query_retrieval as perform_recipes_multi_query_retrieval,
)

# Load environment variables from a .env file
load_dotenv(override=True)

# Set up logging in the app.log file
log = logging.getLogger("assistant")
logging.basicConfig(filename="app.log", level=logging.INFO)

# Import and configure OpenAI
from langchain_openai import ChatOpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

chat_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
#log.info(api_key)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "mysecret")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("SUPABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database setup
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
   id = db.Column(db.Integer, primary_key=True)
   username = db.Column(db.String(150), unique=True, nullable=False)
   email = db.Column(db.String(150), unique=True, nullable=False)
   password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
   return db.session.get(User, int(user_id))

# Initialize Supabase and LangChain components
supabase_https_url = os.getenv("SUPABASE_HTTPS_URL")
supabase_key = os.getenv("SUPABASE_KEY")
 
supabase_client = create_client(supabase_https_url, supabase_key, options=ClientOptions(
   postgrest_client_timeout=120,
   storage_client_timeout=120,
   schema="public",
 ))

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

books_vector_store = SupabaseVectorStore(
    client=supabase_client,
    table_name="books",
    embedding=embeddings,
    query_name="match_books"
    )

recipes_vector_store = SupabaseVectorStore(
    client=supabase_client,
    table_name="recipes_llm",
    embedding=embeddings,
    query_name="match_recipes_llm"
    )

# Define MemorySaver instance for langgraph agent
memory = MemorySaver()


# Create the agent tools for the RAG functions
# Similarity Search (Books)
def create_books_similarity_search_tool():
    @tool
    def get_books_similarity_search(input: str) -> str:
        """
        Tool to perform a simple similarity search on the 'books' vector store.
        Returns the top matching chunks as JSON.
        """
        
        query = input.strip()

        results = perform_books_similarity_search(query, books_vector_store)
        # 'perform_similarity_search' might return Documents or a custom structure.
        # Convert it to JSON or a string
        return json.dumps(results, default=str)
    
    return get_books_similarity_search

# Retrieval QA (Books)
def create_books_retrieval_qa_tool():
    @tool
    def get_books_retrieval_qa(input: str) -> str:
        """
        Tool for short Q&A over the 'books' corpus using retrieval QA.
        """

        query = input.strip()

        chain_result = perform_books_retrieval_qa(query, chat_llm, books_vector_store)
        # Typically returns a dict with 'answer', 'sources', 'source_documents', etc.
        return json.dumps(chain_result, default=str)
    
    return get_books_retrieval_qa

# Similarity Search (Recipes)
def create_recipes_similarity_search_tool():
    @tool
    def get_recipes_similarity_search(input: str) -> str:
        """
        Tool to perform a simple similarity search on the 'recipes_llm' vector store.
        Returns the top matching chunks as JSON.
        """
        
        query = input.strip()

        results = perform_recipes_similarity_search(query, chat_llm, recipes_vector_store)
        # 'perform_recipes_similarity_search' might return Documents or a custom structure.
        # Convert it to JSON or a string
        return json.dumps(results, default=str)
    
    return get_recipes_similarity_search

# Retrieval QA (Recipes)
def create_recipes_self_query_tool():
    @tool
    def get_recipes_self_query(input: str) -> str:
        """
        Tool for searching recipes with metadata-based self-query retrieval. (E.g. filter by recipe_type, cuisine, special_considerations, etc.)
        """

        query = input.strip()

        results = perform_recipes_self_query_retrieval(query, chat_llm, recipes_vector_store, SupabaseVectorTranslator())
        # Typically returns a dict with 'answer', 'sources', 'source_documents', etc.
        return json.dumps(results, default=str)
    
    return get_recipes_self_query

# Multi-Query Retrieval (Recipes)
def create_recipes_multi_query_tool():
    @tool
    def get_recipes_multi_query(input: str) -> str:
        """
        Tool for searching recipes with metadata-based self-query retrieval. (E.g. filter by recipe_type, cuisine, special_considerations, etc.)
        And for using multi-query retrieval to generate multiple versions of the user’s question from different perspectives.
        """

        query = input.strip()

        results = perform_recipes_multi_query_retrieval(query, chat_llm, recipes_vector_store, SupabaseVectorTranslator())
        # Typically returns a dict with 'answer', 'sources', 'source_documents', etc.
        return json.dumps(results, default=str)
    
    return get_recipes_multi_query

# Routes
# Index route
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")  # Serve the chat interface

# Stream route
@app.route("/stream", methods=["GET"])
@login_required
def stream():

    books_retrieval_qa_tool = create_books_retrieval_qa_tool()
    books_similarity_search_tool = create_books_similarity_search_tool()
    recipes_similarity_search_tool = create_recipes_similarity_search_tool()
    recipes_self_query_tool = create_recipes_self_query_tool()
    recipes_multi_query_tool = create_recipes_multi_query_tool()

    system_message_content = """You are ChefBoost, a helpful cooking assistant that 
        provides recipe information and cooking advice from a database.

        When providing recipes, format them in this exact structured way:

        1. Start with "Title: [Recipe Name]" on the first line
        2. Then add "Recipe Type: [Type]" where type is one of: dessert, appetizer, main course, soup, salad, beverage, breakfast, side dish
        3. Then add "Cuisine: [Cuisine]" where cuisine is the origin (e.g., Italian, French, Thai, etc.)
        4. Then add "Special Considerations: [Any dietary notes]" for allergies or diets (e.g., vegetarian, gluten-free, dairy-free)
        5. Then "Ingredients:" followed by a bulleted list (use - for bullets) Always include the amount of each ingredient used in the recipe
        6. Add "Instructions:" followed by numbered steps (use 1. 2. 3. etc.)
        7. ALWAYS add "Source: [Source]" with either the source of the recipe or "ChefBoost AI" if created by you
        8. ALWAYS add "Date: [Date]" with the current date

        Important: 
        - DO NOT use markdown formatting like bold (** **) or respond with JSON, just use plain text when providing the recipe
        - ALWAYS use the exact headings shown above with colons (:)
        - When multiple recipes are requested, create separate recipes with Title: at the start of each
        - Keep each recipe complete with ALL fields
        - When answering non-recipe questions, provide clear, helpful responses as a cooking assistant
        - You have access to tools to search a database of recipes and cooking information

        Example format:
        Title: Italian Tiramisu
        Recipe Type: dessert
        Cuisine: Italian  
        Special Considerations: contains eggs and dairy
        Ingredients:
        - 6 egg yolks
        - 3/4 cup sugar
        - 16 oz mascarpone cheese
        - 1 1/2 cups strong brewed coffee, cooled
        - 24 ladyfinger cookies
        - 1/4 cup cocoa powder for dusting
        Instructions:
        1. Beat egg yolks and sugar until light and fluffy
        2. Fold in mascarpone cheese until smooth
        3. Quickly dip each ladyfinger in coffee and arrange in serving dish
        4. Spread half the mascarpone mixture over ladyfingers
        5. Add another layer of dipped ladyfingers
        6. Top with remaining mascarpone mixture
        7. Dust with cocoa powder and refrigerate for at least 4 hours
        Source: Traditional Italian Cookbook
        Date: 04/06/2025
        """

    graph = create_react_agent(
        model=chat_llm,
        tools=[
            books_retrieval_qa_tool,
            books_similarity_search_tool,
            recipes_similarity_search_tool,
            recipes_self_query_tool,
            recipes_multi_query_tool,
        ],
        checkpointer=memory,
        prompt=system_message_content,
        debug=True
    )

    inputs = {"messages": [("user", request.args.get("query", ""))]}
    config = {"configurable": {"thread_id": "thread-1"}}

    @stream_with_context
    def generate():
        stream_iterator = graph.stream(inputs, config, stream_mode="messages")
        current_node = None
        current_output = ""
        try:
            while True:
                try:
                    msg, metadata = next(stream_iterator)
                except StopIteration:
                    break

                node = metadata.get("langgraph_node")
                # If we detect a change in the node, assume previous output was intermediate.
                if current_node is None:
                    current_node = node
                elif node != current_node:
                    current_node = node
                    current_output = ""  # reset accumulator for new node

                if msg.content:
                    current_output += msg.content

                # Once we get a finish signal for the current node, we break out.
                if metadata.get("finish_reason") == "stop":
                    break
        except GeneratorExit:
            return  # client disconnected
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            return

        # Yield only the final aggregated output from the last node.
        yield f"data: {json.dumps(current_output)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), content_type="text/event-stream")

# Sign up route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
       username = request.form.get("username")
       email = request.form.get("email")
       password = request.form.get("password")
       
       # Query the database and filter for the username
       user = User.query.filter_by(username=username).first()
       
       # If there is a user with the same username in the database, reload the page
       if user:
            flash("Username already registered.", "error")
            return redirect(url_for("signup"))
       
       # If new user, hash their password using werkzeug module
       new_user = User(
           username=username,
           email=email,
           password=generate_password_hash(password, method="pbkdf2:sha256")
       )

       # Add new user to the database
       db.session.add(new_user)
       db.session.commit()
       flash("Account created successfully! Please log in.", "success")
       return redirect(url_for("login"))
    
    return render_template("signup.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
       username = request.form.get("username")
       password = request.form.get("password")
       
       user = User.query.filter_by(username=username).first()
       
       if not user or not check_password_hash(user.password, password):
           flash("Invalid username or password.", "error")
           return redirect(url_for("login"))
       
       login_user(user)
       flash("Logged in successfully!", "success")
       return redirect("/")
    
    return render_template("login.html")

# My Account route
@app.route("/my_account", methods=["GET", "POST"])
@login_required
def my_account():
    if request.method == "POST":
       current_password = request.form.get("current_password")
       new_password = request.form.get("new_password")
       confirm_password = request.form.get("confirm_password")
       if not check_password_hash(current_user.password, current_password):
            flash("Current password is incorrect.", "error")
            return redirect(url_for("my_account"))
       if new_password != confirm_password:
           flash("New passwords do not match.", "error")
           return redirect(url_for("my_account"))
       
       current_user.password = generate_password_hash(new_password, method="pbkdf2:sha256")
       db.session.commit()
       flash("Password updated successfully!", "success")
       return redirect(url_for("index"))
    return render_template("my_account.html", user=current_user)

# Logout route
@app.route("/logout")
@login_required
def logout():
   logout_user()
   flash("You have been logged out.", "success")
   return redirect(url_for("login"))

# Function to add to the log in the app.log file
def log_run(run_status):
    if run_status in ["cancelled", "failed", "expired"]:
        log.error(str(datetime.datetime.now()) + " Run " + run_status + "\n")

# Run the Flask server
if __name__ == "__main__":
    with app.app_context():
       db.create_all()  # Ensure the database is created
       # print("Database created.")
    app.run(debug=True)