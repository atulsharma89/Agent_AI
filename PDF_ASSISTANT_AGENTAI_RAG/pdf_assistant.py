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

knowledge_base=PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection='recipes',db_url=db_url)
)

#load knowledgebase

knowledge_base.load()

# Create storage

storage=PgAssistantStorage(table_name="pdf_assistant",db_url=db_url)

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
