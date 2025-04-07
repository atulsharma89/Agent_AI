#pip install sqlalchemy

import typer 
import dataclasses
from typing import Optional,List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
#from phi.storage.assistant.postgres import PgAssistantStorage
from sqlalchemy.dialects import postgresql

from phi.agent import Agent
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2


from phi.knowledge.pdf import PDFUrlKnowledgeBase
#from phi.vectordb.pgvector import Pgvector2
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

os.environ["PHI_API_KEY"]=os.getenv("PHI_API_KEY")

#https://docs.phidata.com/vectordb/pgvector

#independnt autonomus AI that should be able to assist for different dii tasks

#Also, I used docker to spin up a pgvector-enabled Postgres container:

#docker run -d \
#  -e POSTGRES_DB=ai \
#  -e POSTGRES_USER=ai \
# -e POSTGRES_PASSWORD=ai \
#  -e PGDATA=/var/lib/postgresql/data/pgdata \
#  -v pgvolume:/var/lib/postgresql/data \
#  -p 5532:5432 \
#  --name pgvector \
#  phidata/pgvector:16

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


#https://docs.phidata.com/vectordb/pgvector


#create knowledgebase with all the pdf we gave
#This ingests the PDF and stores its semantic content into the pgvector collection called recipes.

knowledge_base=PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection='recipes',db_url=db_url)
)

#load knowledgebase

knowledge_base.load()

# Create storage
#Assistant Storage Layer
#This sets up persistent storage for assistant sessions using Postgres.

#Maintains Chat History Across Sessions

#Without persistent storage, every time the assistant restarts, it “forgets” past interactions.
#Persistent storage allows the assistant to recall previous conversations, making responses smarter and more personalized over time.
#2️⃣ Supports Long-Running Interactions

#Many workflows require multi-step interactions (e.g., summarizing a document, refining a search, or tracking a task).
#With stored session states, the assistant can pick up where it left off without starting from scratch.
#3️⃣ Enables Multi-User Support

#If you're building for more than one user, storing sessions in Postgres allows you to track each user's session separately.
#You can store metadata like user_id, run_id, and timestamps, making analytics and user experience better.
#4️⃣ Auditing and Debugging

#Keeping logs of past conversations helps in auditing decisions, tracking model performance, and improving user experience.
#It also makes it easier to debug issues or improve prompts based on real usage data.
5️⃣# Scales with Application

#Using Postgres (with SQLAlchemy or similar ORMs) ensures that your application can scale reliably with proper data integrity, backup, and migration support.


storage=PgAssistantStorage(table_name="pdf_assistant",db_url=db_url)

#Assistant Logic

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__=="__main__":
    typer.run(pdf_assistant)
