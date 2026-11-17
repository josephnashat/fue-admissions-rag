### Edges
from langchain_core.messages import  AIMessage
# from utilities.self_rag.hallucination_grader_chain import get_hallucination_grader_chain
from utilities.self_rag.hallucination_and_answer_grader_chain import get_hallucination_and_answer_grader_chain
# from utilities.self_rag.answer_grader_chain import get_answer_grader_chain
from utilities._admin_utils import load_text_question_answer, upload_data
from utilities._globals import GLOBAL_VARIABLES as GlobalVariables
import asyncio

def document_sources(state):
    """Source documents behind the current answer, recorded with the learned entry (safeguard (a))."""
    return [s for d in state.get("documents") or [] for s in (getattr(d, "metadata", None) or {}).get("sources", "").split("|") if s]

async def write_to_pinecone(question, source, answer, doc_sources=None):
    ai_generated_document = load_text_question_answer(question,source, answer, doc_sources)
    vector_store = GlobalVariables.Cached_Question_answer_Vector_Store
    record_manager = GlobalVariables.Cached_Question_answer_Record_Manager
    upload_data(vector_store, record_manager, [ai_generated_document])
    

def decide_to_answer_question(state):
    """
    Determines whether to answer the user question or not.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS Domain Question to Answer---")
    
    relevant_question = state["domain_relevant_question"].lower()
    if relevant_question == "relevant":
        print("---DECISION: RELEVANT QUESTION JUMP TO RETRIEVE DOCUMENTS---")
        operation_mode = state["operation_mode"] if state["operation_mode"] else "Simple"
        if operation_mode == "Simple":       
            asyncio.run(write_to_pinecone(state["naive_generated_question"],"Naive AI Generated DB", state["generation"]))
            # ai_generated_document = load_text_question_answer(state["naive_generated_question"],"Naive AI Generated DB", state["generation"])
            # vector_store = GlobalVariables.Cached_Question_answer_Vector_Store
            # record_manager = GlobalVariables.Cached_Question_answer_Record_Manager
            # upload_data(vector_store, record_manager, [ai_generated_document])
            return "simple mode"    
        else:
            return "relevant_domain_knowledge"
    elif relevant_question == "conversation":
        return "conversation"
    else:
        # IRRELEVANT QUESTION
        print("---DECISION: END DISCUSSION---")       
        return "irrelevant_domain_knowledge"
    
def decide_to_continue_kyc(state):
    print("---Decide to Continue KYC---")
    messages = state["messages"]
    if (isinstance(messages[-1], AIMessage) and messages[-1].tool_calls) or ("applicant_profile" in state and state["applicant_profile"] != None):
        ret = "kyc_done"
    else: 
        ret = "kyc_collecting"

    
    return ret
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    generate_counter = state["generate_counter"]
    max_trial_number = state["max_trial_number"]
    if generate_counter >= max_trial_number: # and len(filtered_documents) > 0:
        print("---DECISION: Can't loop more to transform question---")
        return "best possible transformation"
    else: 
        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "all_docs_are_not_relevant"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "relevant_doc_exist"

def check_ai_generated_db(state):
    if state["answered_before"] and len(state["documents"])> 0:
        return "question_answerd_before"
    else:
        return "new_question"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    documents = state["documents"]
    generate_counter = state["generate_counter"]
    max_trial_number = state["max_trial_number"]
    if state["answered_before"]:
        return "answered before"
    else:
        if generate_counter >=max_trial_number: # and len(documents) > 0:
            print("---DECISION: GENERATION IS NOt GROUNDED IN DOCUMENTS or Hallucination but this is the best possible I can do---")
            # print(documents)
            # a fallback answer with no supporting documents cannot be traced or invalidated, so it is not stored
            if document_sources(state):
                asyncio.run(write_to_pinecone(state["question"],"AI Generated DB", state["generation"], document_sources(state)))
            # ai_generated_document = load_text_question_answer(state["question"],"AI Generated DB", state["generation"])
            # vector_store = GlobalVariables.Cached_Question_answer_Vector_Store
            # record_manager = GlobalVariables.Cached_Question_answer_Record_Manager
            # upload_data(vector_store, record_manager, [ai_generated_document])
            return "best possible answer"
        else: 
            print("---CHECK HALLUCINATIONS---")
            generation = state["generation"]
            question = state["question"]
            score = get_hallucination_and_answer_grader_chain().invoke(
            {"context": documents, "generation": generation,"question": question}
            )
            grade = score.binary_score
            # Check hallucination
            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                print("--START BUILDING SELF KNOWLEDGE CHAIN--")
                asyncio.run(write_to_pinecone(state["question"],"AI Generated DB", state["generation"], document_sources(state)))
                # ai_generated_document = load_text_question_answer(state["question"],"AI Generated DB", state["generation"])
                # vector_store = GlobalVariables.Cached_Question_answer_Vector_Store
                # record_manager = GlobalVariables.Cached_Question_answer_Record_Manager
                # index_result = upload_data(vector_store, record_manager, [ai_generated_document])
                return "perfect answer"
                # Check question-answering
                # print("---GRADE GENERATION vs QUESTION---")
                # score = get_answer_grader_chain().invoke({"question": question, "generation": generation})
                # grade = score.binary_score
                # if grade == "yes":
                #     print("---DECISION: GENERATION ADDRESSES QUESTION---")
                #     print("--START BUILDING SELF KNOWLEDGE CHAIN--")
                #     ai_generated_document = load_text_question_answer(state["question"],"AI Generated DB", state["generation"])
                #     vector_store=insert_or_fetch_embeddings("cached-question-answers",None)
                #     record_manager = create_record_manager("pinecone/cached-question-answers", False)
                #     index_result = upload_data(vector_store, record_manager, [ai_generated_document], cleanup=None)
                #     return "perfect answer"
                # else:
                #     print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                #     # print(generation)
                #     # print('poor answer')
                #     return "poor answer"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "hallucination-poor answer"