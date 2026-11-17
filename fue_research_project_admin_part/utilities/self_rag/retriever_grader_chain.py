def get_retrieval_grader_chain():
    from langchain_core.prompts import ChatPromptTemplate
    # from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score and summary for relevance check on retrieved documents."""

        consolidated_text: str = Field(
            description="A consolidated representation of the concatenated documents, preserving all information."        )
        
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = GlobalVariables.Chat_OpenAI_temp_7
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt for summarization and grading
    system = """


### **Role**:  
You are an assistant tasked with evaluating the relevance of retrieved documents to a user question.

---

### **Instructions**:

#### **Step 1: Consolidate Documents**
- Combine all provided documents into a **single, comprehensive, and coherent text** that captures all relevant details.
- Include every significant point from the documents without summarizing or omitting critical information.
- Expand and elaborate on key points to ensure the consolidated text is full and informative.
- Pay special attention to metadata, particularly for questions related to **gallery, images, or videos**, and integrate this information seamlessly into the consolidated text.
- Ensure the final consolidated text is detailed, coherent, and aligned with the user question.

#### **Step 2: Assess Relevance**
- Based on the consolidated text, evaluate whether it is relevant to the user question using the following criteria:
  - **Relevant**: If the text contains keywords or semantic meaning directly addressing the user question.
  - **Not Relevant**: If the text is unrelated or only tangentially related to the user question.
- Assign a **binary relevance score**:
  - *Yes* for relevant.
  - *No* for not relevant.

---

### **Output Requirements**:
1. **Consolidated Text**:  
   - A single, well-organized document containing all details from the provided documents.  
   - Include all critical points, expanded where necessary, and ensure coherence and alignment with the user question.  

2. **Relevance Score**:  
   - A binary score (`yes` or `no`) indicating whether the consolidated text directly addresses the user question.

---

        
        """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", 
             "Documents List: \n\n {documents} \n\n User question: {question}"),
        ]
    )

    return grade_prompt | structured_llm_grader



# def get_retrieval_grader_chain():
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_openai import ChatOpenAI
#     from pydantic import BaseModel, Field

#     # Data model
#     class GradeDocuments(BaseModel):
#         """Binary score for relevance check on retrieved documents."""

#         binary_score: str = Field(
#             description="Documents are relevant to the question, 'yes' or 'no'"
#         )


#     # LLM with function call
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     structured_llm_grader = llm.with_structured_output(GradeDocuments)

#     # Prompt
#     system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
#         It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#         If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
#         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
#     grade_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
#         ]
#     )

#     return grade_prompt | structured_llm_grader


#     # question = "fue"
#     # docs = retriever.invoke(question)
#     # doc_txt = docs[1].page_content
#     # print(retrieval_grader.invoke({"question": question, "document": doc_txt}))