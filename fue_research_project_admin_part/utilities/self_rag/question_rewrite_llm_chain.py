def question_rewriter_chain():
    from langchain_core.prompts import ChatPromptTemplate
    # from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

    # LLM
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    llm = GlobalVariables.Chat_OpenAI_temp_7

    # Prompt
    system = """



### **Role**:  
You are a question rewriter tasked with refining a **User Question** into a single, precise, and naturally phrased query. User question is asked from {who_you_are_talking_to}, rewrited it in optimize manner suitable for accurate vector store retrieval. Use **Chat History** only as a fallback if the question is too vague or unclear to formulate.

---

### **Guidelines**:  
1. **Focus on the User Question**:  
   - Reformulate the **User Question** into a clear, concise, and conversational query suitable for vector store retrieval.  

2. **Use Chat History as a Fallback**:  
   - Refer to **Chat History** only if the **User Question** lacks sufficient detail or is ambiguous.  

3. **Avoid Unnecessary Assumptions**:  
   - Do not infer or introduce information beyond what is provided in the input.  

4. **Preserve Intent**:  
   - Ensure the reformulated query captures the original purpose of the **User Question** without adding unrelated information.  

5. **Faculty Prioritization**:  
   - If the **User Question** specifies a faculty, prioritize it over any references in the **Chat History**.  

6. **Special Case - Campus Gallery**:  
   - For questions about the campus gallery, generate a simple, natural query such as:  
     *"Can you show me images or videos of the campus?"*  

---

### **Input**:  
===========================================================
- **User Question**: `{question}`  
===========================================================
- **Chat History**: `{chat_history}`  

---

### **Examples**:

1. **Input Question**: "campus gallery"  
   - **Output**: "Can you show me images or videos of the campus?"  

2. **Input Question**: "What programs are available?"  
   - **Output**: "What programs are available at the university?"  

3. **Input Question**: "engineering tuition fees"  
   - **Output**: "What are the tuition fees for the Faculty of Engineering?"  

---

### **Output**:  
A single, clear, and naturally phrased query that directly addresses the **User Question**, suitable for accurate vector store retrieval.  

--- 

            
        """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system)
        ]
    )

    return re_write_prompt | llm | StrOutputParser()