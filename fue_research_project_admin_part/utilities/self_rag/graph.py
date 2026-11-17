
from utilities.self_rag.graph_state import GraphState
from langgraph.graph import END, StateGraph
from utilities.self_rag.nodes import kyc, domain_agent, retrieve, grade_documents, generate, transform_query, generate_conversation
from utilities.self_rag.edges import decide_to_answer_question,decide_to_continue_kyc, decide_to_generate, grade_generation_v_documents_and_question,check_ai_generated_db

# docker volume create postgres_data
# docker run -d --name postgres_container -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=postgres -p 5432:5432 -v C:\docker\postgres_data:/var/lib/postgresql/data postgres:15







def build_graph():    
    DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    workflow = StateGraph(GraphState)
    # Define the nodes
    workflow.add_node("kyc", kyc)  # kyc
    workflow.add_node("domain_knowledge", domain_agent)  # domain_knowledge_agent
    # workflow.add_node("conversation", generate_conversation)  # conversation
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae


    # Build graph
    workflow.set_entry_point("kyc")
    # workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "transform_query",
        check_ai_generated_db,
        {
            "question_answerd_before": "generate",
            "new_question": "retrieve"
        },
    )

    workflow.add_conditional_edges(
        "domain_knowledge",
        decide_to_answer_question,
        {
            "irrelevant_domain_knowledge": END,
            "relevant_domain_knowledge": "transform_query",
            "conversation": END,
            "simple mode": END
        },
    )
    workflow.add_conditional_edges(
        "kyc",
        decide_to_continue_kyc,
        {
            "kyc_done": "domain_knowledge",
            "kyc_collecting": END
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    # workflow.add_edge("conversation", END)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "all_docs_are_not_relevant": "transform_query",
            "relevant_doc_exist": "generate",
            "best possible transformation": "generate"
        },
    )

    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "hallucination-poor answer": "generate",
            # "hallucination": "generate",
            # "poor answer": "generate",
            "perfect answer": END,
            "best possible answer": END,
            "answered before": END
        },
    )
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    pool =  ConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    )
    postgres_checkpointer = PostgresSaver(pool)
    # postgres_checkpointer.setup()
    return workflow.compile(checkpointer=postgres_checkpointer)
    
# Compile
# memory = MemorySaver()

# app = workflow.compile(checkpointer=memory)
# from IPython.display import Image, display

# try:
#     display(Image(app.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass


# def init_graph():
#     from langgraph.checkpoint.postgres import PostgresSaver
#     # docker volume create postgres_data
#     # docker run -d --name postgres_container -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=postgres -p 5432:5432 -v C:\docker\postgres_data:/var/lib/postgresql/data postgres:15

#     DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
#     connection_kwargs = {
#         "autocommit": True,
#         "prepare_threshold": 0,
#     }

#     from psycopg_pool import ConnectionPool

#     with ConnectionPool(
#         conninfo=DB_URI,
#         max_size=20,
#         kwargs=connection_kwargs,
#     ) as pool:
#         postgres_checkpointer = PostgresSaver(pool)
#         postgres_checkpointer.setup()

