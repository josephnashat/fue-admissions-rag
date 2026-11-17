
def get_domain_grader_chain():
  from langchain_core.prompts import ChatPromptTemplate
  # from langchain_openai import ChatOpenAI
  from pydantic import BaseModel, Field
  from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

  # Data model
  class GradeDomainQuestion(BaseModel):
      """Question Category as Relevant, Irrelevant or conversation as well as the Questino Answer from the provided context"""

      question_category: str = Field(
          description="relevant, irrelevant, or conversation based on Question asked"
      )
      reformulated_user_question: str = Field(
          description="reformulated user question suitable for vectorstore retrieval"
      )
      question_answer: str = Field(
        description="Proposed answer to the user question."
    )


  llm = GlobalVariables.Chat_OpenAI_temp_0
  structured_llm_grader = llm.with_structured_output(GradeDomainQuestion)
  system = """

  **Task**: You are talking to {who_you_are_talking_to} , and your task is to Classify the user’s question into Relevant, Irrelevant, or Conversation, and answer the user question based on the provided context, your answer should be in {answer_in}.
  
  Steps:
    1. Translate User Question to English if it is provided in a different language.
    
    2. Use the provided Chat History to clearly identify user intent and user question.

    3. Categorize the Question:
        - Relevant: The question can be answered based on the provided context.
        - Irrelevant: The question is outside the provided context.
        - Conversation: The question includes greetings, thank-yous, or casual interactions.

    4. Answer user question:
        - For Relevant: Generate answer for user query based on the provided context, answer should be formatted into bullets or paragraphs.
        - For Irrelevant: Politely explain that the question is outside the scope and offer to assist with related queries, in the same language.
        - For Conversation: Respond in a polite, friendly tone, in the same language.

---

### **Input:**

**User Question**: `{question}`  
===========================================================
**Context**: `{context}`  
===========================================================
**Chat History**: `{chat_history}`
===========================================================

---

### **Output Format (JSON):**

```json
{{
    "reformulated_user_question": "<Reformulated User Question>",
    "question_category": "<Relevant | Irrelevant | Conversation>",
    "question_answer": "<Your Answer in {answer_in}>"
}}
```

---

### **Notes for GPT-4o Mini**:
- For vague questions, **map keywords** to relevant, meaningful queries from the perspective of a university applicant.  
- Ensure responses are clear, polite, and context-aware.  





"""

  domain_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system),
          # ("human", "User question: {question}, context: {context}"),
      ]
  )

  return domain_prompt | structured_llm_grader


if __name__ == "__main__":
  question = "who wins US election?"
  context = "Future University in Egypt offers programs like Engineering, Dentistry, and Pharmacy."
  user_intent = "Applicant Ahmed is interested in studying Computer Science at Future University in Egypt."

  response = get_domain_grader_chain().invoke({"question": question, "context": context, "who_you_are_talking_to": user_intent})
  print(response)