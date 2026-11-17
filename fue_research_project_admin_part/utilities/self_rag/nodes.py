from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import Document
from utilities.self_rag.kyc_llm_chain import get_kyc_chain,get_messages_info
from utilities.self_rag.domain_evaluator_chain import get_domain_grader_chain
# from utilities.self_rag.retriever import get_retriever,get_chroma_retriever
from utilities.self_rag.answer_generation_chain import get_rag_chain
from utilities.self_rag.conversation_chain import get_conversation_chain
from utilities.self_rag.retriever_grader_chain import get_retrieval_grader_chain
from utilities.self_rag.question_rewrite_llm_chain import question_rewriter_chain
# from utilities.self_rag.retriever_ai import get_retriever_ai_qa_chain
from utilities._globals import GLOBAL_VARIABLES as GlobalVariables
from utilities._admin_utils import current_admission_cycle


def format_user_data(sb):
    import json
    if type(sb) == str:
        obj = json.loads(sb)
    else:
        obj = sb
    # print('format user data: ', obj)
    if obj:
        return obj, f"You are interacting with {obj['name']} an undergraduate student interested in joining Future University. Applicant primary interest is in the Faculty of {obj['faculty']} , though applicant may consider other faculties in their inquiries."
    return None, "You are talking to undergraduate student interested to join Future University"

def kyc(state):
    print("---KYC---")
    operation_mode = state["messages"][-1].metadata["mode"] if "mode" in state["messages"][-1].metadata else "Simple"
    if "applicant_profile" not in state:        
        messages = get_messages_info([m.content for  m in state['messages']  ])
        response = get_kyc_chain().invoke(messages)
        last_question = state["messages"][-1].content

        if isinstance(response, AIMessage) and response.tool_calls:
            sb = response.additional_kwargs["tool_calls"][0]["function"]["arguments"]
            obj, _ = format_user_data(sb)
            return {"applicant_profile": obj ,"generation":"", "operation_mode": operation_mode}
        else:    
            return {"question":last_question, "messages": [response], "generation": response.content, "operation_mode": operation_mode}
    else:
         return {"question":state["messages"][-1].content, "operation_mode": operation_mode}


def domain_agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide whether it belongs to FUE domain or not

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL DOMAIN AGENT--- ")
 
    # pastQuestions = ', '.join ([m.content for  m in state['messages'] if isinstance(m, HumanMessage) ][-4:]) 
    chat_history = ''
    for  m in state['messages'][-4:]:
        if isinstance(m, HumanMessage):
            chat_history = chat_history + "Human Asked: " +  m.content
        else:
            chat_history = chat_history + "AI Answered: " +  m.content
    
    lastQuestion = [m.content for  m in state['messages'] if isinstance(m, HumanMessage) ][-2:]
    answer_in = detect_language(lastQuestion[-1])
    # documents = get_chroma_retriever().invoke(pastQuestions)
    documents = GlobalVariables.Chroma_Retriever.invoke(lastQuestion[-1])
    stuffed_docs = ''
    for doc in documents:
        stuffed_docs = stuffed_docs + doc.page_content + " \n\n" + "metadata: " + ', '.join(doc.metadata) + " \n\n\n\n"
    _, applicant_profile = format_user_data(state["applicant_profile"])
    
    response = get_domain_grader_chain().invoke({"question":lastQuestion[-1], "context": stuffed_docs, "chat_history": chat_history, "answer_in": answer_in, "who_you_are_talking_to": applicant_profile})
    generation = response.question_answer
    # if response.question_category == 'irrelevant':
    #     generation = "I'm sorry, but I can only assist with questions about Future University in Egypt (FUE) and its programs."
        
    return {"question":lastQuestion[-1],"naive_generated_question": response.reformulated_user_question, "domain_relevant_question": response.question_category, "generation":generation , "generate_counter": 0, "max_trial_number": 3, "documents": [], "filtered_docs_count": 0, "check_kb": True, "answered_before": False}



def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    old_documents = state["documents"]
    # Retrieval
    # documents = get_retriever().invoke(question)
    documents =GlobalVariables.Pinecone_KB_Retriever.invoke(question)
    if documents:
        return {"documents": documents, "question": question}
    else:
        return {"documents": old_documents, "question": question}
    
def detect_language(input_text):
    import re
    # Step 1: Sanitize input to keep only alphabetic characters
    sanitized_text = re.sub(r'[^A-Za-zأ-ي]', '', input_text)

    # Step 2: Check for Arabic and English letters
    contains_arabic = bool(re.search(r'[أ-ي]', sanitized_text))

    # Step 3: Determine response language
    if contains_arabic:
        return "Arabic"
    else:
        return "English"


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    _, applicant_profile = format_user_data(state["applicant_profile"])
    lastQuestions = [m.content for  m in state['messages'] if isinstance(m, HumanMessage) ][-1:]
    answer_in = detect_language(' '.join(lastQuestions))
    if documents:
        generation = get_rag_chain().invoke({"context": documents, "question": question, "answer_language": answer_in, "who_you_are_talking_to":applicant_profile})
        return {"documents": documents, "question": question, "generation": generation, "generate_counter": state["generate_counter"] + 1}
    else:
        return {"documents": documents, "question": question, "generate_counter": state["generate_counter"] + 1}
        



def generate_conversation(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE CONVERSATION---")
    question = state["question"]
    # RAG generation
    _, applicant_profile = format_user_data(state["applicant_profile"])
    generation = get_conversation_chain().invoke({"question": question, "who_you_are_talking_to":applicant_profile})
    return {"question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    reduced_docs = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents if hasattr(doc,"page_content") and hasattr(doc,"metadata") ]
    result = get_retrieval_grader_chain().invoke(
                {"question": question, "documents": reduced_docs}
            )    
    # documents.append(Document(page_content=result.consolidated_text))
    if result.binary_score == "yes":
        # keep the source documents so a learned answer can be traced back to them (safeguard (a))
        sources = sorted({d["metadata"].get("source", "") for d in reduced_docs} - {""})
        return {"documents": [Document(page_content=result.consolidated_text, metadata={"sources": "|".join(sources)})], "question": question}
    else:
        return {"documents":[], "question": question}
    # filtered_docs_count = state["filtered_docs_count"]      
    # # Score each doc
    # # print(filtered_docs_count)
    # filtered_docs = []
    # docs_to_add = []
    # filtered_cnt = 0
    # for i, d in enumerate(documents):
    #     # print(i, d)
    #     if i >= filtered_docs_count - 1:
    #         # print('checking doc', filtered_cnt)
    #         filtered_cnt += 1
    #         score = get_retrieval_grader_chain().invoke(
    #             {"question": question, "document": d.page_content}
    #         )
    #         grade = score.binary_score
    #         if grade == "yes":
    #             print("---GRADE: DOCUMENT RELEVANT---")
    #             filtered_docs.append(d)
    #         else:
    #             print("---GRADE: DOCUMENT NOT RELEVANT---")
    #             # print(d.page_content)
    #             continue
    # if filtered_docs_count == 0:
    #     docs_to_add = filtered_docs
    # else:
    #     docs_to_add = documents + filtered_docs
            
    # return {"documents": docs_to_add, "question": question, "filtered_docs_count": len(docs_to_add)}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    # idx = state["generate_counter"] + 1
    pastQuestions = [m.content for  m in state['messages'] if isinstance(m, HumanMessage) ][-6:] # + "," + question
    # Re-write question
    _, applicant_profile = format_user_data(state["applicant_profile"])
    better_question = question_rewriter_chain().invoke({"question": pastQuestions[-1], "chat_history":pastQuestions , "who_you_are_talking_to":applicant_profile })
    if state["check_kb"]:        
        # documents = get_retriever_ai_qa_chain().invoke(better_question)
        documents = GlobalVariables.Pinecone_KB_AI_Retriever.invoke(better_question)
        # discard naive entries (safeguard (b)) and entries from earlier admission cycles (safeguard (c))
        cycle = current_admission_cycle()
        documents = [doc for doc in documents if "source" in doc.metadata and 'Naive' not in doc.metadata["origin"]
                     and doc.metadata.get("cycle") == cycle]
        # documents[0].metadata['answer']      
        # discard naive learnt documents          
        if documents:            
            return {"question": better_question , "documents": [Document(page_content=doc.metadata["answer"]) for doc in documents], "answered_before": True, "check_kb": False}        
    #add more to generate counter every time the query is transformed to limit infinite looping
    return {"question": better_question, "generate_counter": state["generate_counter"]+1 ,"answered_before": False, "check_kb": False}
