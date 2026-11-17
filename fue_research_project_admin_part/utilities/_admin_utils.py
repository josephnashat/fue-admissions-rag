
# from langchain.indexes import SQLRecordManager, index
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.pgvector import PGVector
# from langchain_community.document_loaders import DirectoryLoader
from importlib import metadata
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from sympy import content
from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

index_name = 'future-university-in-egypt'
namespace = f"pinecone/{index_name}"
# embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

def get_openai_embeddngs():
    # from langchain_openai import OpenAIEmbeddings
    # return OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    return GlobalVariables.Open_AI_Embedding
# region machine learning helpers
#*********Functions for dealing with Model related tasks...************

#Read dataset for model creation
def read_data(data):
    import pandas as pd
    df = pd.read_csv(data,delimiter=',', header=None)  
    return df

#Create embeddings instance
def get_embeddings():
    # from langchain.embeddings import SentenceTransformerEmbeddings
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # return embeddings
    return GlobalVariables.Sentence_Transformer_Embedding

#Generating embeddings for our input dataset
def create_embeddings(df,embeddings):
    import numpy as np
    # df[3] = df[0].apply(lambda x: embeddings.embed_query(x))
    # df[3] = df[1].apply(lambda x: list(str(x)))
    df[3]= embeddings.embed_documents(df[0].tolist())
    # Combine embeddings with scores for training
    scores = df[1].tolist()
    X = np.hstack([df[3].tolist(), np.array(scores).reshape(-1, 1)])  # Concatenate embeddings and scores
    Y = df[2].tolist()
    return X, Y

#Splitting the data into train & test
def split_train_test__data(X,Y):
    from sklearn.model_selection import train_test_split
    # Split into training and testing sets
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
    print(len(sentences_train))
    return sentences_train, sentences_test, labels_train, labels_test

#Get the accuracy score on test data
def get_score(svm_classifier,sentences_test,labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score


#Region for Langchain and Pinecone specific tasks

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):    
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        # print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        # print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def load_document_with_image(docx_file):
    """
    Extract text and images from a .docx file.
    Returns full text and a list of image paths.
    """
    from docx import Document
    import re
    import cloudinary.uploader

    # Configure Cloudinary
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    )
    # os.path.join(os.path.dirname(__file__), 'images_in_doc')
    IMAGE_SAVE_DIR = 'fue_research_project_admin_part/images_in_doc'
    # IMAGE_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'images_in_doc')
    
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    doc = Document(docx_file)
    full_text = ""
    image_paths = []
    
    # Extract text
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            full_text += paragraph.text + "\n"

    # Extract images
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            img = rel.target_part.blob
            image_name = os.path.basename(rel.target_ref)
            image_path = os.path.join(IMAGE_SAVE_DIR, image_name)
            # Upload the image
            response = cloudinary.uploader.upload(image_path)
            # Save the image
            with open(image_path, "wb") as f:
                f.write(img)
            image_paths.append(response["secure_url"])

        # Define a regex pattern for YouTube links
    youtube_pattern = r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+"
    # List to store found YouTube links
    youtube_links = []
    # Iterate through all paragraphs in the document
    for paragraph in doc.paragraphs:
        # Search for YouTube links in the text of the paragraph
        matches = re.findall(youtube_pattern, paragraph.text)
        youtube_links.extend(matches)


    return full_text, image_paths, youtube_links

def recursive_chunking_with_images(text, image_paths, youtubelinks):
    """
    Chunk text recursively and associate images based on their proximity.
    Returns a list of LangChain Document objects.
    """
   
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    doc = Document(page_content=text)
    chunks = text_splitter.split_documents([doc])

    # Map images to chunks based on proximity
    documents = []
    images_per_chunk = len(image_paths) // max(1, len(chunks))  # Distribute images evenly
    remaining_images = len(image_paths) % max(1, len(chunks))

    youtube_per_chunk = len(youtubelinks) // max(1, len(chunks))  # Distribute youtube evenly
    remaining_youtube = len(youtubelinks) % max(1, len(chunks))



    for i, chunk in enumerate(chunks):
        start_idx = i * images_per_chunk
        end_idx = start_idx + images_per_chunk
        if end_idx == len(chunks) and remaining_images > 0:
            chunk_images = image_paths[start_idx:]
        else:
            chunk_images = image_paths[start_idx:end_idx]

        start_idx2 = i * youtube_per_chunk
        end_idx2 = start_idx2 + youtube_per_chunk
        if end_idx2 == len(chunks) and remaining_youtube > 0:
            chunk_youtube = youtubelinks[start_idx:]
        else:
            chunk_youtube = youtubelinks[start_idx2:end_idx2]

        # Create a LangChain Document with metadata
        documents.append(
            Document( 
                page_content=chunk.page_content ,  # Adjust based on your chunker output
                # metadata={"images": chunk_images, "youtube": chunk_youtube, "source":"fue_research_project_admin_part/images_in_doc" }
                metadata={"images": ', '.join(chunk_images), "youtube": ', '.join(chunk_youtube), "source":"fue_research_project_admin_part/images_in_doc" }
                
            )
        )


    return documents


def semantic_chunking_with_images(text, image_paths, youtubelinks, embedding, for_chroma=False, source='fue_research_project_admin_part/images_in_doc'):
    """
    Chunk text semantically and associate images based on their proximity.
    Returns a list of LangChain Document objects.
    """
    # Use SemanticChunker to split text
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain.schema import Document
    
    text_splitter = SemanticChunker(embedding, min_chunk_size=600)
    doc = Document(page_content=text)
    chunks = text_splitter.split_documents([doc])

    # Map images to chunks based on proximity
    documents = []
    images_per_chunk = len(image_paths) // max(1, len(chunks))  # Distribute images evenly
    remaining_images = len(image_paths) % max(1, len(chunks))

    youtube_per_chunk = len(youtubelinks) // max(1, len(chunks))  # Distribute youtube evenly
    remaining_youtube = len(youtubelinks) % max(1, len(chunks))



    for i, chunk in enumerate(chunks):
        start_idx = i * images_per_chunk
        end_idx = start_idx + images_per_chunk
        if end_idx == len(chunks) and remaining_images > 0:
            chunk_images = image_paths[start_idx:]
        else:
            chunk_images = image_paths[start_idx:end_idx]

        start_idx2 = i * youtube_per_chunk
        end_idx2 = start_idx2 + youtube_per_chunk
        if end_idx2 == len(chunks) and remaining_youtube > 0:
            chunk_youtube = youtubelinks[start_idx:]
        else:
            chunk_youtube = youtubelinks[start_idx2:end_idx2]

        if for_chroma:
            # Create a LangChain Document with metadata
            documents.append(
                Document( 
                    page_content=chunk.page_content ,  # Adjust based on your chunker output
                    # metadata={"images": chunk_images, "youtube": chunk_youtube, "source":"fue_research_project_admin_part/images_in_doc" }
                    metadata={"images": ','.join(chunk_images), "youtube": ','.join(chunk_youtube), "source":source }
                    
                )
            )
        else:
            # Create a LangChain Document with metadata
            documents.append(
                Document( 
                    page_content=chunk.page_content ,  # Adjust based on your chunker output
                    # metadata={"images": chunk_images, "youtube": chunk_youtube, "source":"fue_research_project_admin_part/images_in_doc" }
                    metadata={"images": chunk_images, "youtube": chunk_youtube, "source":source }
                    
                )
            )


    return documents

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document_URL(urls, in_one_doc=False):
    from langchain.schema import Document
    from langchain_community.document_loaders import WebBaseLoader   
    docs = [WebBaseLoader(url).load() for url in urls]
    if in_one_doc:
        docs_list = [item.page_content for sublist in docs for item in sublist]    
        return Document(page_content= ''.join(docs_list), metadata={"source": ' - '.join(urls)})
    else:
        docs_list = [item for sublist in docs for item in sublist]
        return docs_list

def current_admission_cycle():
    """Admission cycle label, e.g. '2026/2027'. Override with the ADMISSION_CYCLE environment variable."""
    from datetime import date
    if os.getenv("ADMISSION_CYCLE"):
        return os.getenv("ADMISSION_CYCLE")
    today = date.today()
    # a new cycle opens in June, when applications for the next academic year start
    start = today.year if today.month >= 6 else today.year - 1
    return f"{start}/{start + 1}"

def load_text_question_answer(question, source, answer, doc_sources=None):
    # Metadata (optional)
    from langchain.schema import Document

    metadata = {
        "source":  question,
        "answer": answer,
        "origin": source,
        # safeguard (c): entries are reused only within the admission cycle that produced them
        "cycle": current_admission_cycle(),
        # safeguard (a): documents the answer was built from, so the entry can be invalidated when they change
        "doc_sources": "|".join(sorted(set(doc_sources or [])))
    }
    document = Document(page_content=question)
    document.page_content = question
    document.metadata = metadata
    return document


def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def chunk_data_semantic(data, embedding):
    from langchain_experimental.text_splitter import SemanticChunker
    text_splitter = SemanticChunker(embedding, min_chunk_size=1500)
    chunks = text_splitter.split_documents(data)
    return chunks

def insert_or_fetch_embeddings(index_name, chunks):
    # importing the necessary libraries and initializing the Pinecone client
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import ServerlessSpec

    pc = pinecone.Pinecone()
        
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    embeddings = GlobalVariables.Open_AI_Embedding


    # loading from existing index
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index 
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
        ) 
        )

        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object. 
        if chunks:
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        else:
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
        
    return vector_store

def get_pinecone_chunk_by_id(chunk_id ,index_name_str:str):
    
   print(chunk_id, index_name_str)
   import pinecone
   if "/" in index_name_str:
       return pinecone.Pinecone().Index("future-university-in-egypt").fetch(ids=[chunk_id])
   return pinecone.Pinecone().Index("cached-question-answers").fetch(ids=[chunk_id]) 


def get_pinecone_chunk_by_id_list(chunk_ids ,index_name_str:str):
    
   import pinecone
   if "AI Generated DB" in index_name_str:
       return pinecone.Pinecone().Index("cached-question-answers").fetch(chunk_ids)    
   return pinecone.Pinecone().Index("future-university-in-egypt").fetch(chunk_ids)
    
def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')
  
def create_record_manager(namespace, init=False):
    from langchain.indexes import SQLRecordManager, index   
    record_manager = SQLRecordManager(
        namespace, db_url=DB_URI #"sqlite:///record_manager_cache.sql"
    )
    if init:
        record_manager.create_schema()
    return record_manager



def invalidate_learned_entries(sources=None):
    """Safeguard (a): delete learned answers built from any of the given document sources.
    With sources=None, delete every learned answer that was built from documents (manual entries are kept)."""
    import pinecone
    targets = set(sources or [])
    index = pinecone.Pinecone().Index("cached-question-answers")
    stale = []
    for ids in index.list():
        for vid, vec in index.fetch(ids=list(ids)).vectors.items():
            used = set(filter(None, (vec.metadata or {}).get("doc_sources", "").split("|")))
            if used and (sources is None or used & targets):
                stale.append(vid)
    if stale:
        index.delete(ids=stale)
        GlobalVariables.Cached_Question_answer_Record_Manager.delete_keys(stale)
    return len(stale)

def upload_data(vector_store, record_manager, chunks, cleanup="incremental",source_id_key="source"):
    from langchain.indexes import index
    return index(
        chunks,
        record_manager,
        vector_store,
        cleanup=cleanup,
        source_id_key=source_id_key,
    )
  
#region PostGres specific    
def get_db_connection():
    # --CREATE EXTENSION IF NOT EXISTS pgcrypto;
    # CREATE TABLE Users (
    #     id SERIAL PRIMARY KEY,              -- Unique identifier for each user
    #     username VARCHAR(255) NOT NULL UNIQUE, -- User's login username
    #     password_hash TEXT NOT NULL,        -- Hashed password
    #     created_at TIMESTAMP DEFAULT NOW()  -- Timestamp for when the user is created
    # );

    # INSERT INTO Users (username, password_hash)
    # VALUES ('joe', crypt('joe', 'fue_salt'));

    import psycopg2
    from psycopg2 import sql
    return psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password=os.environ.get("POSTGRES_PASSWORD", "postgres")
    )

            
def fetch_data(query):
    import pandas as pd
    try:
        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query)
        
        # Fetch column names and data
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to a list of dictionaries (for AgGrid)
        # return [dict(zip(columns, row)) for row in data]
        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        return pd.DataFrame()    
    


#Rgion helpers for system

@dataclass
class FlattenedCheckpoint:
    thread_id: str
    step: int
    source: str
    key_step: Optional[str] = None
    question: Optional[str] = None
    generation: Optional[str] = None
    document_sources: List[str] = field(default_factory=list)
    document_page_contents: List[str] = field(default_factory=list)
    transform_query_content: Optional[str] = None
    applicant_profile: Optional[Dict[str, Any]] = None

queries_list = {'all_user_meta_data': """
                SELECT metadata FROM public.checkpoints where thread_id = %s order by checkpoint_id desc                
                """,
                'only_kyc': """
                select metadata from checkpoints where metadata->>'writes' like '%"kyc"%' and metadata->>'writes' like '%"applicant_profile"%' 
                """,
                'all_user_messages': """
                SELECT  ARRAY_AGG(distinct metadata->'writes'->'generate'->>'question')  FROM public.checkpoints where 
                metadata->>'writes' like '%"generate"%' and 
                thread_id = '{}'
                """
                }

def load_chat_profiles(str_query, param1=None) -> List[FlattenedCheckpoint]:
    from psycopg2 import sql
    conn = get_db_connection()
    cur = conn.cursor()
    query = sql.SQL(str_query)
    if param1:
        cur.execute(query, (param1,))
    else:
        cur.execute(query)
    result = cur.fetchall()
    print(query)
    cur.close()
    conn.close()
    checkpoints = [r[0] for r in result]
    flattened_checkpoints = []

    for c in checkpoints:
        thread_id = c['thread_id']
        step = c['step']
        source = c['source']
        writes = c.get('writes', {})
        
        key_step = None
        question = None
        generation = None
        document_sources = []
        document_page_contents = []
        transform_query_content = None
        applicant_profile = None

        if writes:
            generations = None
            if 'generate' in writes:
                key_step = 'generate'
                generations = writes['generate']
            elif 'grade_documents' in writes:
                key_step = 'grade_documents'
                generations = writes['grade_documents']
            elif 'retrieve' in writes:
                key_step = 'retrieve'
                generations = writes['retrieve']
            elif 'transform_query' in writes:
                key_step = 'transform_query'
                generations = writes['transform_query']
            elif 'domain_knowledge' in writes:
                key_step = 'domain_knowledge'
                generations = writes['domain_knowledge']
            elif 'kyc' in writes:
                key_step = 'kyc'
                generations = writes['kyc']
                temp_profile= generations.get('applicant_profile', None)               
                if temp_profile is not None:
                    applicant_profile = generations.get('applicant_profile', None)
                    applicant_profile['user_id'] = thread_id 

            if generations:
                generation = generations.get('generation')
                question = generations.get('question')
                
                if key_step != 'transform_query':
                    documents = generations.get('documents', [])
                    for doc in documents:
                        kwargs = doc.get('kwargs', {})
                        metadata = kwargs.get('metadata', {})
                        document_sources.append(metadata.get('source', ''))
                        document_page_contents.append(kwargs.get('page_content', ''))
                else:
                    transform_query_content = generations.get('messages', {}).get('kwargs', {}).get('content', None)

                # Create a FlattenedCheckpoint object
                flattened_checkpoint = FlattenedCheckpoint(
                    thread_id=thread_id,
                    step=step,
                    source=source,
                    key_step=key_step,
                    question=question,
                    generation=generation,
                    document_sources=document_sources,
                    document_page_contents=document_page_contents,
                    transform_query_content=transform_query_content,
                    applicant_profile=applicant_profile
                )
                flattened_checkpoints.append(flattened_checkpoint)

    return flattened_checkpoints

def persist_applicants(df) :
#     CREATE TABLE IF NOT EXISTS public.applicants
# (
#     name text COLLATE pg_catalog."default",
#     email text COLLATE pg_catalog."default",
#     mobile text COLLATE pg_catalog."default",
#     chat_rank bigint,
#     faculty text COLLATE pg_catalog."default",
#     user_id integer NOT NULL,
#     prediction text COLLATE pg_catalog."default",
#     CONSTRAINT user_id PRIMARY KEY (user_id)
# )
    from sqlalchemy import create_engine
    engine = create_engine(DB_URI)
    # Persist DataFrame to PostgreSQL
    # df.to_sql(
    #     name="applicants",       # Name of the table in the database
    #     con=engine,              # Database connection
    #     if_exists="append",     # What to do if the table exists: 'fail', 'replace', 'append'
    #     index=False              # Whether to write the DataFrame index as a column
    # )
    # Insert or update values
    insert_query = """
    INSERT INTO applicants (name, email, mobile, faculty,user_id, chat_rank, prediction)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (user_id) DO UPDATE
    SET name = EXCLUDED.name, email = EXCLUDED.email, mobile = EXCLUDED.mobile
    , faculty = EXCLUDED.faculty, chat_rank = EXCLUDED.chat_rank, prediction = EXCLUDED.prediction;
    """

    # Use psycopg2 connection to execute the query
    with engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            for row in df.itertuples(index=False):
                cursor.execute(insert_query, row)
        conn.connection.commit()
 
# Check login credentials
def check_credentials(username, password):
#     CREATE TABLE IF NOT EXISTS public.users
# (
#     id integer NOT NULL DEFAULT nextval('users_id_seq'::regclass),
#     username character varying(255) COLLATE pg_catalog."default" NOT NULL,
#     password_hash text COLLATE pg_catalog."default" NOT NULL,
#     created_at timestamp without time zone DEFAULT now(),
#     CONSTRAINT users_pkey PRIMARY KEY (id),
#     CONSTRAINT users_username_key UNIQUE (username)
# )

    from psycopg2 import sql
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = sql.SQL("""SELECT *
                            FROM Users
                            WHERE username = %s
                            AND password_hash = crypt(%s, 'fue_salt')
                        """)
        cur.execute(query, (username, password))
        result = cur.fetchone()
        print(query)
        cur.close()
        conn.close()
        return result is not None
    except Exception as e:
        # st.error(f"Error connecting to database: {e}")
        return False
    
def load_perssisted_chat_messages():
    from sqlalchemy import create_engine
    import pandas as pd    
    engine = create_engine(DB_URI)
    # Load data from PostgreSQL into a DataFrame
    query = "SELECT * FROM applicants;"
    return pd.read_sql(query, con=engine)
def query_chat_messages(user_id):
    from psycopg2 import sql
    conn = get_db_connection()
    cur = conn.cursor()
    str_query = queries_list['all_user_messages']
    str_query = str_query.format(user_id)
    query = sql.SQL(str_query)
    cur.execute(query)
    result = cur.fetchall()
    print(query)
    cur.close()
    conn.close()
    return result

def get_chat_messages(user_id):
    result = query_chat_messages(user_id)
    if result is None:
        return 0
    return calculate_chat_rank(result[0][0])

# Define the function to calculate "chat_rank"
def calculate_chat_rank(question):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    # Data model
    class RankedChat(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        chat_rank: str = Field(...,
            description="Scale of 1 to 10 (where 1 means highly unlikely and 10 means highly likely)"
        )


    # LLM with function call
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(RankedChat)

    # Prompt
    system = """
    You are evaluating an applicant's likelihood of applying to an undergraduate degree program at a university based on the list of questions they have asked. Carefully assess their level of interest by analyzing the depth, relevance, and specificity of their questions. Consider factors such as whether their questions indicate genuine interest in the program details, curiosity about academic opportunities, or practical concerns like fees and deadlines. Using a scale of 1 to 10 (where 1 means highly unlikely and 10 means highly likely), assign a score based on the overall strength of their intent to apply. Take a moment to reason and weigh their likelihood before finalizing your decision. Provide only the numerical scale in your response, without any explanation or justification.

    Output: [Scale from 1 to 10]
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Input:[List of applicant questions]: \n\n {questions} "),
        ]
    )

    chat_rank_grader = grade_prompt | structured_llm_grader
    return chat_rank_grader.invoke({"questions": question}).chat_rank

def rag_algorithm_retriever():
    # importing the necessary libraries and initializing the Pinecone client
    INDEX_NAME = "future-university-in-egypt"
    from langchain_community.vectorstores import Pinecone
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    embeddings = GlobalVariables.Open_AI_Embedding
    vector_store = Pinecone.from_existing_index(INDEX_NAME, embeddings)        
    return vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.6})
    

def rag_algorithm_used(question):
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")
    system = """You are an intelligent student recruiter online support agent who is dedicated to assisting prospective university applicants throughout their enrollment process.

    Utilize the provided contextual information to respond to the user's inquiry to the best of your ability.  

    Maintain a polite tone, express gratitude for their question, and ensure your responses are attractive, humanized conversation and convincing to inspire enthusiasm and encourage the applicant to join the university. Don't add signatures or any other information that is not relevant to the conversation.
    Try to generate your answers in an organized way in bullets for better understanding, remember that you are talking to teenager.


    """
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Context: \n\n {context} \n\n User question: {question}"),
        ]
    )

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

    # Chain
    retriever = rag_algorithm_retriever()
    from operator import itemgetter

    retrieval_chain = (
        {
            "context": (lambda x: x["question"]) | retriever,
            # "question": lambda x: x["question"],
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return retrieval_chain.invoke({"question": question})


def answer_question_about_video(question, context):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    # Data model
    # LLM with function call
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    # Prompt
    system = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Make your answer organized and listed as bullets as much as you can
    Question: {question} 
    Context: {context} 
    Answer:
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system)
        ]
    )

    rag_chain = grade_prompt | llm
    return rag_chain.invoke({"question": question, "context": context}).content
   
def predict_applicant(row):
    result = query_chat_messages(row['user_id'])
    if result is None or result[0] is None or result[0][0] is None:
        return '---'
    
    embeddings= get_embeddings()
    question = ', '.join(result[0][0])
    query_result = embeddings.embed_query(question)
    score = row['chat_rank']
    import numpy as np
    query_result = np.hstack([query_result, str(score)])
    import joblib
    
    Fitmodel = joblib.load('fue_research_project_admin_part/modelsvm.pk1')
    # Fitmodel = joblib.load(os.path.join(os.path.dirname(__file__), 'modelsvm.pk1'))
    
    result=Fitmodel.predict([query_result])
    return result[0]


def rank_applicants_chat(df):
    # Apply the function to the "user_id" column
    df["chat_rank"] = df["user_id"].apply(get_chat_messages)
    return df
    
def predict_applicants(df, model_exist):
    if model_exist == 0:
        df["prediction"] = "----"
    else:
        df["prediction"] = df.apply(predict_applicant, axis=1)
    return df   

def persist_client_chat(userid, msgsList):
#     CREATE TABLE IF NOT EXISTS public.chat_sessions
# (
#     uuid character varying COLLATE pg_catalog."default" NOT NULL,
#     applicant_id integer,
#     add_date timestamp without time zone DEFAULT now(),
#     chat_messages character varying COLLATE pg_catalog."default",
#     CONSTRAINT chat_sessions_pkey PRIMARY KEY (uuid)
# )
    from sqlalchemy import create_engine
    engine = create_engine(DB_URI)
    with engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            cursor.execute("update chat_sessions  set chat_messages = %s where uuid = %s",(msgsList,userid))
        conn.connection.commit()

    # pass

def add_new_chat_channel(gen_uuid):
    final_gen_uuid = ''
    if gen_uuid :
        final_gen_uuid = gen_uuid
    else:
        import uuid
        final_gen_uuid =  str(uuid.uuid4())
    from sqlalchemy import create_engine
    engine = create_engine(DB_URI)
    with engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            cursor.execute("insert into chat_sessions (uuid) values ('{}')".format(final_gen_uuid))
        conn.connection.commit()
    return final_gen_uuid
        
def get_chat_channel():
    import uuid
    gen_uuid = str(uuid.uuid4())
    from psycopg2 import sql
    conn = get_db_connection()
    cur = conn.cursor()
    query = sql.SQL("select uuid from chat_sessions order by add_date desc limit 1")
    cur.execute(query)
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        gen_uuid = result[0]
    else:
        add_new_chat_channel(gen_uuid)
    return gen_uuid

def get_all_chat_channels():
    # return fetch_data("select uuid from chat_sessions")
    return fetch_data("""
                      SELECT chat_sessions.uuid thread_id,
                    CONCAT(
                        metadata->'writes'->'kyc'->'applicant_profile'->>'name', ' | ',
                        metadata->'writes'->'kyc'->'applicant_profile'->>'email', ' | ',
                        metadata->'writes'->'kyc'->'applicant_profile'->>'mobile', ' | ',
                        metadata->'writes'->'kyc'->'applicant_profile'->>'faculty',' | ', chat_sessions.uuid
                    ) AS channel_name
                FROM checkpoints right join  chat_sessions on checkpoints.thread_id = chat_sessions.uuid
                and  metadata->>'writes' LIKE '%"kyc"%' 
                and metadata->>'writes' LIKE '%"applicant_profile"%' 
                      """)

def get_chat_channel_by_id(uuid):
    from psycopg2 import sql
    conn = get_db_connection()
    cur = conn.cursor()
    query = sql.SQL("select chat_messages from chat_sessions where chat_messages is not null and uuid = '{}'".format(uuid))
    cur.execute(query)
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

def kyc_reached(uuid):
    from psycopg2 import sql
    conn = get_db_connection()
    cur = conn.cursor()
    query = sql.SQL("""
                   SELECT *,metadata->'writes'->'kyc'->'applicant_profile'->>'name'
                    FROM checkpoints 
                    where  metadata->>'writes' LIKE '%"kyc"%' 
                    and metadata->>'writes' LIKE '%"applicant_profile"%' 
                    and checkpoints.thread_id = '{}'""".format(uuid))
    cur.execute(query)
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

    
def chroma_db_for_domain_rag(rebuild=False):
    import os
    from langchain.vectorstores import Chroma
    # from langchain.embeddings import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    embeddings = GlobalVariables.Open_AI_Embedding
    # Path to the folder containing .docx files
    folder_path = "fue_research_project_admin_part/documents"

    # Directory to persist Chroma database
    persist_directory = "fue_research_project_admin_part/chroma_db"

    if os.path.exists(persist_directory) and rebuild:
        # documents changed: drop the domain-evaluation index so it is rebuilt from the current documents
        Chroma(persist_directory=persist_directory, embedding_function=embeddings).delete_collection()
        type(GlobalVariables).Chroma_Retriever = None   # class attribute: re-created lazily on next use
    if os.path.exists(persist_directory) and not rebuild:
        chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return chroma_db
    else:
        # Step 1: Load all .docx files
        def load_docx_files(folder):
            docx_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.docx') or f.endswith('.pdf')] 
            docx_files.append("fue_research_project_admin_part/images_in_doc/gallery.docx")
            docs = []
            alltext = []
            for file in docx_files:
                name, extension = os.path.splitext(file)
                if extension == '.docx':
                    text, images, youtubeLinks = load_document_with_image(file)
                else:
                    text = ' ------- '.join([d.page_content for d in load_document(file)])
                    images = []
                    youtubeLinks = []
                alltext.append(text)
                if len(images) > 0:
                    text = text + " Gallery/Image Links: " + ' - '.join(images)
                if len(youtubeLinks) > 0:     
                    text = text + " Youtube Links: " + ' - '.join(youtubeLinks)
            
            chunks = semantic_chunking_with_images('\n\n '.join(alltext),[],[],embeddings, True,'consolidated_docs')
            docs.extend(chunks)
            return docs

        # Load documents
        chunks = load_docx_files(folder_path)

        # Create a new persistent Chroma DB
        chroma_db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

        print("Chroma database successfully set up.")
        return chroma_db
    
def query_read_count(question):
    from psycopg2 import sql
    conn = get_db_connection()
    cur = conn.cursor()
    str_query = """
                SELECT thread_id,metadata,metadata->'writes'->'generate'->>'question' AS question
                FROM checkpoints
                WHERE metadata->'writes'->'generate'->>'question' ILIKE  %s
                AND metadata->'writes'->'generate'->>'question' IS NOT NULL;
                """
    # str_query = str_query.format(question)
    # query = sql.SQL(str_query)
    # cur.execute(query)
    cur.execute(str_query, (f"%{question}%",))  # Use %% for wildcard matching with ILIKE
    result = cur.fetchall()
    print(str_query)
    cur.close()
    conn.close()
    return result
  
def get_read_count(row):
    result = query_read_count(row['question'])
    if result is None:
        return 0
    else:
        return len(result)    

def persist_ai_qa(df):
#     CREATE TABLE IF NOT EXISTS public.ai_generated_qa
# (
#     uuid character varying COLLATE pg_catalog."default" NOT NULL,
#     source character varying COLLATE pg_catalog."default",
#     question character varying COLLATE pg_catalog."default",
#     answer character varying COLLATE pg_catalog."default",
#     read_count integer,
#     CONSTRAINT ai_generated_qa_pkey PRIMARY KEY (uuid)
# )
    df["read_count"] = df.apply(get_read_count, axis=1)
    from sqlalchemy import create_engine
    engine = create_engine(DB_URI)
    insert_query = """INSERT INTO ai_generated_qa (uuid, source, question, answer, read_count)
    VALUES (%s,%s, %s, %s, %s)
    ON CONFLICT (uuid) DO UPDATE
    SET question = EXCLUDED.question,source = EXCLUDED.source, answer = EXCLUDED.answer, read_count = EXCLUDED.read_count
    """
    # Use psycopg2 connection to execute the query
    with engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            cursor.execute("delete from ai_generated_qa")
        conn.connection.commit()

    with engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            for row in df.itertuples(index=False):
                cursor.execute(insert_query, row)
        conn.connection.commit()
    