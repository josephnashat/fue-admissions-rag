import streamlit as st
from utilities._admin_utils import *
import os
import pandas as pd
from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

@st.cache_resource
def load_model():
    import whisper
    model = whisper.load_model("base")
    return model

# Callback function for audio recording
def toggle_recording():
    if st.session_state.is_recording:
        st.session_state.is_recording = False
    else:
        st.session_state.is_recording = True

def main():
    
    st.set_page_config(page_title="Managing Data...", layout="wide")
    st.title("Managing Data...📁 ")
    
    # Create tabs
    tab_titles = ['Upload Docs', 'Upload URL', 'Video POC',  "Docs with Images", "View Docs and Images","Delete Files"]
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        st.header('Uploading Files')
        st.write('Upload docx, pdf, or txt files')
        # Upload the pdf file...
        docs_uploader = st.file_uploader("Only PDF, Docx, Txt files are allowed", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="docs_uploader")
        # Extract the whole text from the uploaded pdf file
        if docs_uploader is not None and len(docs_uploader) > 0:
            documents = []
            for i,f in enumerate(docs_uploader):                
                file_path = os.path.join('fue_research_project_admin_part/documents/',f.name)
                with st.spinner('Wait for it...'):
                    st.write(f"👉Loading " + f.name)
                    documents.append(load_document(file_path))
            st.write(f"👉Reading Documents done")
            document = [item for sublist in documents for item in sublist]
            with st.spinner('Wait for it...'):
                # Create chunks
                docs_chunks=chunk_data_semantic(document, get_openai_embeddngs())
                #st.write(docs_chunks)
                st.write("👉Splitting data into chunks done")
                # Create the embeddings
                # vector_store=insert_or_fetch_embeddings(index_name,None)
                vector_store=GlobalVariables.Future_University_Vector_Store
                st.write("👉Creating embeddings instance done")
                #Generate the record manager to use to track documents
                # record_manager = create_record_manager(namespace, False)
                record_manager = GlobalVariables.Future_University_Record_Manager
                #Upload chunks to store
                upload_data(vector_store, record_manager, docs_chunks)
                # safeguard (a): learned answers built from these documents are now stale
                removed = invalidate_learned_entries({c.metadata.get("source", "") for c in docs_chunks})
                st.write(f"👉Invalidated {removed} learned answers that depended on these documents")
                chroma_db_for_domain_rag(rebuild=True)   # keep the domain-evaluation index in step with the documents
                st.write("👉Domain-evaluation index rebuilt")
            st.success("Successfully pushed the embeddings to Pinecone")
    with tabs[1]:
        urlList = [
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/admissions_steps",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/admission_requirements/general_requirements",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/admission_requirements/english_language_entry_requirements",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/admission_requirements/computer_skills_placement_test",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/admission_requirements/faculty_required_subjects",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/admission_requirements/documents_and_entry_requirements",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/registration_policy",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/requirements_for_transfer_students",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/scholarships",
            "https://www.fue.edu.eg/admissions/undergraduate_applicants/tuition_policy"
            ]
        st.header('Uploading URLs')
        st.write('Upload multiple URLs to the vector store, one per line')
        urls = st.text_area("Enter multiple URLs (one per line):", value='\n'.join(urlList))
        btn_upload_urls = st.button("Upload URLs", key="btn_upload_urls")
        if btn_upload_urls:
            with st.spinner('Wait for it...'):
                # vector_store=insert_or_fetch_embeddings(index_name,None)
                vector_store = GlobalVariables.Future_University_Vector_Store
                st.write("👉Creating embeddings instance done")
                #Generate the record manager to use to track documents
                # record_manager = create_record_manager(namespace, False)    
                record_manager = GlobalVariables.Future_University_Record_Manager
                document = load_document_URL(urls.split('\n'))
                st.write("👉Reading Documents done")                           
                # chunks = chunk_data_semantic([document], get_openai_embeddngs())
                # st.write("👉Chunking Documents done")                           
                get_openai_embeddngs()
                upload_data(vector_store, record_manager, document)
                removed = invalidate_learned_entries({d.metadata.get("source", "") for d in document})
                st.write(f"👉Invalidated {removed} learned answers that depended on these pages")
                chroma_db_for_domain_rag(rebuild=True)   # keep the domain-evaluation index in step with the documents
                st.write("👉Domain-evaluation index rebuilt")
            st.success("Successfully pushed the embeddings to Pinecone")
    with tabs[2]:
        if 'btn_answer_video_question_state' not in st.session_state:
            st.session_state['btn_answer_video_question_state'] = []
        st.header('Uploading Youtube URLs - just POC')
        st.write('''
                 Upload URL for Youtube video to the vector store \n
                  samples: \n
                  https://www.youtube.com/watch?v=SQUjZFx0t4k \n
                  https://www.youtube.com/watch?v=-hifcHTu714 \n
                  https://www.youtube.com/watch?v=4nxGjtpVNQU \n
                  https://www.youtube.com/watch?v=f6yrEfpoCTg''')
        youtube_url = st.text_input("Enter Youtube URL:")
        btn_upload_youtube_urls = st.button("Parse Video", key="btn_upload_youtube_urls")
        if btn_upload_youtube_urls:
            with st.spinner('Wait for the magic...'):               
                import yt_dlp
                import imageio_ffmpeg
                import uuid
                downloaded_file_name = f"./youtube/youtube_{str(uuid.uuid4())}"
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"FFmpeg Path: {ffmpeg_path}")
                st.write("👉AI Model loading done")
                video_url = youtube_url
                ydl_opts = {
                    "format": "bestaudio/best",
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                    }],
                    "outtmpl": downloaded_file_name,
                    "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe()  # Use the installed ffmpeg
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                st.write("👉Downloading video done")
                # Transcribe audio using Whisper
                model = load_model()
                st.write("👉Speech to text done")
                result = model.transcribe(downloaded_file_name+".mp3") 
                # Extract segments with timestamps
                transcription_with_metadata = []
                for segment in result["segments"]:
                    transcription_with_metadata.append({
                        "start_time": segment["start"],  # Start time in seconds
                        "end_time": segment["end"],      # End time in seconds
                        "text": segment["text"]          # Transcribed text
                    })
                # transcription_with_metadata
                st.session_state['btn_answer_video_question_state']=transcription_with_metadata
                st.write("👉Ready to go")                 
        video_question = st.text_input("Ask me anything:", value="summarize")  
        if len(st.session_state['btn_answer_video_question_state']) > 0:
            btn_answer_video_question = st.button("Answer", key="btn_answer_video_question") 
            if btn_answer_video_question:
                with st.spinner('Wait for the magic...'):
                    video_response = answer_question_about_video(video_question, st.session_state['btn_answer_video_question_state'])
                st.write(video_response)
                
                # vector_store=insert_or_fetch_embeddings(index_name,None)
                # st.write("👉Creating embeddings instance done")
                # #Generate the record manager to use to track documents
                # record_manager = create_record_manager(namespace, False)   
                # file_path = downloaded_file_name + '.txt'
                # with open(file_path, 'w') as f:
                #     f.write(result["text"])
                # document = load_document(file_path)
                # st.write("👉Reading Documents done")                                         
                # upload_data(vector_store, record_manager, document)
            st.success("Ready to go...")
    with tabs[3]:
        st.header('Uploading Documents With Images')
        st.write('Upload docx with images')
        # Upload the pdf file...
        single_doc_uploader = st.file_uploader("Only Docx files are allowed", type=["docx"], accept_multiple_files=False, key="single_doc_uploader")       
        if single_doc_uploader:                 
            file_path = os.path.join('fue_research_project_admin_part/images_in_doc/',single_doc_uploader.name)
            with st.spinner('Wait for it...'):
                # print( file_path)
                text, images, youtube = load_document_with_image(file_path)
                st.write("👉Reading Document and Extracting Images Done")
                # Create chunks
                docs_chunks=semantic_chunking_with_images(text, images, youtube, get_openai_embeddngs())
                st.write("👉Splitting data into chunks done")
                # Create the embeddings
                # vector_store=insert_or_fetch_embeddings(index_name,None)
                vector_store=GlobalVariables.Future_University_Vector_Store
                st.write("👉Creating embeddings instance done")
                #Generate the record manager to use to track documents
                # record_manager = create_record_manager(namespace, False)
                record_manager = GlobalVariables.Future_University_Record_Manager
                #Upload chunks to store
                upload_data(vector_store, record_manager, docs_chunks)
                # safeguard (a): learned answers built from these documents are now stale
                removed = invalidate_learned_entries({c.metadata.get("source", "") for c in docs_chunks})
                st.write(f"👉Invalidated {removed} learned answers that depended on these documents")
                chroma_db_for_domain_rag(rebuild=True)   # keep the domain-evaluation index in step with the documents
                st.write("👉Domain-evaluation index rebuilt")
            st.success("Successfully pushed the embeddings to Pinecone")        
    with tabs[4]:
        st.header('Original - Upserted - Chunked Documents')
        st.write('List of Original Documents')
        load_data_button = st.button('Load Data', key='load_data_button')
        if load_data_button or 'show_all_docs_controls' in st.session_state:
            st.session_state.show_all_docs_controls = True
            # # Filter only .docx files
            # directory_path = 'fue_research_project_admin_part/documents'
            # files = os.listdir(directory_path)
            # docx_files = [f for f in files if f.endswith('.docx')]
            # # Prepare list for file data
            # file_data = []       
            # for file_name in docx_files:
            #     file_path = os.path.join(directory_path, file_name)
            #     if os.path.isfile(file_path):  # Check if it's a file
            #         try:
            #             # Read .docx file content
            #             content = load_document(file_path)[0].page_content
            #             file_data.append({"File Name": file_name, "Content": content})
            #         except Exception as e:
            #             file_data.append({"File Name": file_name, "Content": f"Error reading file: {e}"})
            # # Display the table
            # if file_data:
            #     st.dataframe(pd.DataFrame(file_data))
            # else:
            #     st.warning("No .docx files found in the directory.")
            # st.divider()    
            st.write("Images")
            image_dir='fue_research_project_admin_part/images_in_doc'
            image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
            num_columns = len(image_paths)
            columns = st.columns(num_columns)
            for idx, image_path in enumerate(image_paths):
                # Select a column to display the image
                with columns[idx % num_columns]:
                    st.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
            st.divider()    
            st.write('List of upserted records through the indexing')
            df = fetch_data("select key,group_id,to_timestamp(updated_at) from upsertion_record where group_id != 'AI Generated DB' order by to_timestamp(updated_at) desc")   
            if df.empty:
                st.warning("No data returned from the query.")
            else:
                st.dataframe(df)
            st.divider()    
            st.write('List of chunks added to VectorStore')
            load_document_pinecone = st.selectbox("Show details for document_key: ", df["key"] + "#" + df["group_id"], key="load_document_pinecone", index=0)
            if load_document_pinecone:
                with st.spinner('Wait for it...'):  
                    chunk_detail = get_pinecone_chunk_by_id(load_document_pinecone.split("#")[0], load_document_pinecone.split("#")[1])
                    if 'vectors' in chunk_detail:
                        st.write(chunk_detail['vectors'][load_document_pinecone.split("#")[0]]['metadata'])
                   
    with tabs[5]:
        st.header('Deleting Files')
        st.write('Take care you are going to clear all the datastore')
        btn_create_rec = st.button("Create Record Manager", key="btn_create_rec")
        if btn_create_rec:
            with st.spinner('Working on deletion...'):
                record_manager = create_record_manager(namespace, True)
                vector_store=insert_or_fetch_embeddings(index_name,None)
                
                # record_manager =GlobalVariables.Future_University_Record_Manager
                st.success("Successfully Creating Schema")
        btn_clear_db = st.button("🗑️ Delete All Records", key="btn_clear_db")
        if btn_clear_db:
            with st.spinner('Working on deletion...'):
                record_manager = create_record_manager(namespace, False)                
                vector_store=insert_or_fetch_embeddings(index_name,None)
                # record_manager = GlobalVariables.Future_University_Record_Manager
                # vector_store=GlobalVariables.Future_University_Vector_Store
                st.write("👉Deleting internal cache...")
                upload_data(vector_store,record_manager, [],"full")
                st.write("👉Invalidating learned answers built from documents...")
                invalidate_learned_entries()
                st.write("👉Deleting PineCone...")
                delete_pinecone_index(index_name)
                st.success("Successfully cleared the DB")
    




if __name__ == '__main__':
    main()
