from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.embeddings import SentenceTransformerEmbeddings


class GlobalConfig:
    """Class to hold static global variables."""
    # Static variables
    Open_AI_Embedding = None
    Chat_OpenAI_temp_7 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    Chat_OpenAI_temp_0 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # Current_Mode = "Simple" # Simple, Advanced
    Chroma_Retriever = None
    Pinecone_KB_AI_Retriever = None
    Pinecone_KB_Retriever = None


    Sentence_Transformer_Embedding = None
    Cached_Question_answer_Vector_Store = None
    Cached_Question_answer_Record_Manager = None
    Future_University_Vector_Store = None
    Future_University_Record_Manager = None
    index_name = 'future-university-in-egypt'
    namespace = f"pinecone/{index_name}"
    DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

    @staticmethod
    def initialize():
        """Initialize the static variables at startup."""
        GlobalConfig.Open_AI_Embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
        # GlobalConfig.Chat_OpenAI_temp_7 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        # GlobalConfig.Chat_OpenAI_temp_0 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)        
        # GlobalConfig.Sentence_Transformer_Embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        GlobalConfig.Cached_Question_answer_Vector_Store = GlobalConfig.insert_or_fetch_embeddings("cached-question-answers",None)
        GlobalConfig.Cached_Question_answer_Record_Manager = GlobalConfig.create_record_manager("pinecone/cached-question-answers", False)        
        GlobalConfig.Future_University_Vector_Store = GlobalConfig.insert_or_fetch_embeddings("future-university-in-egypt",None)
        GlobalConfig.Future_University_Record_Manager = GlobalConfig.create_record_manager("pinecone/future-university-in-egypt", False)       
        GlobalConfig.Chroma_Retriever = GlobalConfig.get_chroma_retriever()
        GlobalConfig.Pinecone_KB_AI_Retriever = GlobalConfig.Cached_Question_answer_Vector_Store.as_retriever(search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.9,'k': 4})
        GlobalConfig.Pinecone_KB_Retriever = GlobalConfig.Future_University_Vector_Store.as_retriever(search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.6,'k': 4})

    def __getattribute__(self, name):
        # Check if the attribute is None and reinitialize
        value = super().__getattribute__(name)
        if value is None:
            if name == "Open_AI_Embedding":
                GlobalConfig.Open_AI_Embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

            elif name == "Sentence_Transformer_Embedding":
                GlobalConfig.Sentence_Transformer_Embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            elif name == "Cached_Question_answer_Vector_Store":
                GlobalConfig.Cached_Question_answer_Vector_Store = GlobalConfig.insert_or_fetch_embeddings("cached-question-answers",None)

            elif name == "Cached_Question_answer_Record_Manager":
                GlobalConfig.Open_AI_Embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
                GlobalConfig.Cached_Question_answer_Record_Manager = GlobalConfig.create_record_manager("pinecone/cached-question-answers", False)        

            elif name == "Future_University_Vector_Store":
                GlobalConfig.Future_University_Vector_Store = GlobalConfig.insert_or_fetch_embeddings("future-university-in-egypt",None)

            elif name == "Future_University_Record_Manager":
                GlobalConfig.Open_AI_Embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
                GlobalConfig.Future_University_Record_Manager = GlobalConfig.create_record_manager("pinecone/future-university-in-egypt", False)       

            elif name == "Chroma_Retriever":
                GlobalConfig.Open_AI_Embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
                GlobalConfig.Chroma_Retriever = GlobalConfig.get_chroma_retriever()

            elif name == "Pinecone_KB_AI_Retriever":
                GlobalConfig.Open_AI_Embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
                GlobalConfig.Cached_Question_answer_Vector_Store = GlobalConfig.insert_or_fetch_embeddings("cached-question-answers",None)
                GlobalConfig.Pinecone_KB_AI_Retriever = GlobalConfig.Cached_Question_answer_Vector_Store.as_retriever(search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.9,'k': 4})

            elif name == "Pinecone_KB_Retriever":
                GlobalConfig.Open_AI_Embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
                GlobalConfig.Future_University_Vector_Store = GlobalConfig.insert_or_fetch_embeddings("future-university-in-egypt",None)
                GlobalConfig.Pinecone_KB_Retriever = GlobalConfig.Future_University_Vector_Store.as_retriever(search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.6,'k': 4})

        # Return the value after initialization
        value = super().__getattribute__(name)
        return value
    
    def insert_or_fetch_embeddings(index_name, chunks):
        # importing the necessary libraries and initializing the Pinecone client
        import pinecone
        from langchain_community.vectorstores import Pinecone
        from langchain_openai import OpenAIEmbeddings
        from pinecone import ServerlessSpec

        pc = pinecone.Pinecone()
            
        # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
        if GlobalConfig.Open_AI_Embedding is None:
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
        else:    
            embeddings = GlobalConfig.Open_AI_Embedding


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
    
    def create_record_manager(namespace, init=False):
        from langchain.indexes import SQLRecordManager, index   
        record_manager = SQLRecordManager(
            namespace, db_url=GlobalConfig.DB_URI #"sqlite:///record_manager_cache.sql"
        )
        if init:
            record_manager.create_schema()
        return record_manager
    

    def get_chroma_retriever():
        from langchain_community.vectorstores import Chroma
        if GlobalConfig.Open_AI_Embedding is None:
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
        else:    
            embeddings = GlobalConfig.Open_AI_Embedding

        # embeddings = GlobalConfig.Open_AI_Embedding
        persist_directory = "fue_research_project_admin_part/chroma_db"
        chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # return chroma_db.as_retriever(  search_type="mmr",search_kwargs={'k': 6})
        return chroma_db.as_retriever()
    
    

# Singleton instance
GLOBAL_VARIABLES = GlobalConfig()