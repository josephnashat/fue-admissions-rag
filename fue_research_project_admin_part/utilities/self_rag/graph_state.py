
from typing import Annotated, Sequence,List
from numpy import number
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    """Instructions on how to prompt the LLM."""
        
    name: str =  Field(...,
        description="Name of student applying"
    )
    email: str =  Field(...,
        description="Email of student applying, it should be valid email address"
    )
    mobile: str=  Field(...,
        description="Mobile of student applying, it shoudl be saved in international format"
    )
    faculty: str=  Field(...,
        description="Faculty of interest for student, it should be one of the following (Business, Computer, Engineering, Pharmacy, Dentistry, Political Science)"
    )
    

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        domain_relevant_question: whether the question is relevant to the domain knowledge or not
        generate_counter: how many times the LLM has been called to generate an answer or transform query
        messages: list of messages communicated
        max_trial_number: maximum number of trials to generate an answer, default value to compare with the generate counter
        filtered_docs_count: number of documents already filtered out, to start counting after this number in the messages list to check validity of new documents only without going through all the documents again
    """

    question: str
    naive_generated_question: str
    operation_mode: str
    generation: str
    documents: List[str]        
    domain_relevant_question: str
    generate_counter: number
    messages: Annotated[Sequence[BaseMessage], add_messages]
    max_trial_number: number
    filtered_docs_count: number
    applicant_profile: str
    answered_before: bool
    check_kb:bool
    