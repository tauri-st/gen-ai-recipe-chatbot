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

# Define MemorySaver instance for langgraph agent
memory = MemorySaver()


# Routes
# Index route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Serve the chat interface

# Stream route
@app.route("/stream", methods=["GET"])
def stream():

    graph = create_react_agent(
        model=chat_llm,
        tools=[],
        checkpointer=memory,
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

# Function to add to the log in the app.log file
def log_run(run_status):
    if run_status in ["cancelled", "failed", "expired"]:
        log.error(str(datetime.datetime.now()) + " Run " + run_status + "\n")

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)