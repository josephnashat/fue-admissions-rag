def get_hallucination_and_answer_grader_chain():
    from langchain_core.prompts import ChatPromptTemplate
    # from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

    # Data model
    class GradeAnswer(BaseModel):
        """Binary score indicating whether the LLM-generated answer is both grounded in the facts and resolves the question."""
        binary_score: str = Field(
            description="Answer addresses the question and is grounded in the facts, 'yes' or 'no'"
        )

    # LLM with function call
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Deterministic output
    llm = GlobalVariables.Chat_OpenAI_temp_0
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Define the system prompt combining groundedness and resolution
    system = """You are a grader tasked with evaluating an LLM-generated answer based on the following combined criteria:

    Groundedness: Assess whether the answer is supported by and consistent with a provided set of retrieved facts.
    Resolution: Assess whether the answer addresses and resolves the user's question, either partially or fully.

    Provide a single binary score ('yes' or 'no') to indicate whether the answer meets both criteria.

    'Yes': The answer is grounded in the facts and resolves the question.
    'No': The answer fails to meet either or both criteria."""

    # Define the ChatPromptTemplate
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n Retrieved Facts: {context} \n\n LLM generation: {generation}"),
        ]
    )

    # Return the grader chain
    return answer_prompt | structured_llm_grader
