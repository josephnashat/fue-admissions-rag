import streamlit as st
from utilities._admin_utils import *
import os
import pandas as pd
from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

def main():
    
    st.set_page_config(page_title="AI Generated DB", layout="wide")
    st.title("Auto Generated QA...✨ ")
    
    # Create tabs
    tab_titles = ['Learned QA','Manipulate QA']
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        st.write('List of Learned Questions')
        load_qa_data_button = st.button('Load Data', key='load_qa_data_button')
        if load_qa_data_button or ('qa_loaded' in st.session_state  and len(st.session_state['qa_loaded']) > 0):
            with st.spinner("Loading..."):
                df = fetch_data("select key,group_id,to_timestamp(updated_at) from upsertion_record where group_id not ILIKE '%/%' order by to_timestamp(updated_at) desc") 
                if len(df)>0:
                    chunk_details = get_pinecone_chunk_by_id_list(df["key"].tolist(),'AI Generated DB')
                    df_qa_ai = pd.DataFrame(columns=["id", "source", "question", "answer"])
                    for vector in chunk_details["vectors"]:
                        question = chunk_details['vectors'][vector]["metadata"]["text"]
                        answer = chunk_details['vectors'][vector]["metadata"]["answer"]
                        source = chunk_details['vectors'][vector]["metadata"]["origin"]
                        df_qa_ai = pd.concat([df_qa_ai, pd.DataFrame([{"id": vector, "source": source, "question": question, "answer": answer}])], ignore_index=True)
                    st.dataframe(df_qa_ai)
                    if len(df_qa_ai) > 0:
                        st.session_state["qa_loaded"] = 'loaded'
                        btn_persist_and_analyze = st.button("Persist and Analyze Data", key="btn_persist_and_analyze")        
                        if btn_persist_and_analyze:
                            with st.spinner("Persisting..."):
                                persist_ai_qa(df_qa_ai)
                                st.session_state["qa_loaded"] = ''
                            st.success("Data Persisted.")
                else:
                    st.error("Nothing learnt yet.")
    with tabs[1]:        
        st.write('List of Persisted QA')
        load_persisted_qa_data_button = st.button('Load Data', key='load_persisted_qa_data_button')
        if load_persisted_qa_data_button or 'show_per_qa' in st.session_state:
            with st.spinner("Loading..."):
                st.session_state["show_per_qa"] = True
                persisted_qa_df = fetch_data("select uuid, source, question, answer, read_count from ai_generated_qa") 
                st.dataframe(persisted_qa_df)
            st.divider()
            cols = st.columns(2)
            with cols[0]:
                #Add New
                with st.form(key='add_form'):
                    addQ = st.text_input("Question",key="addQ")
                    addA = st.text_area("Answer",key="addA")
                    aaB = st.form_submit_button(label="Add")
                if aaB:
                    with st.spinner(""):
                        ai_generated_document = load_text_question_answer(addQ,"AI Generated DB - Manual", addA)
                        vector_store = GlobalVariables.Cached_Question_answer_Vector_Store
                        record_manager = GlobalVariables.Cached_Question_answer_Record_Manager
                        upload_data(vector_store, record_manager, [ai_generated_document])
                        st.success("New Question Added")
            with cols[1]:
                sel = st.selectbox("Questions:", persisted_qa_df["question"])
                with st.spinner(""):
                    with st.form(key='update_form'):
                        selected_row = persisted_qa_df[persisted_qa_df["question"] == sel]
                        updQ = st.text_input("Question", key="updQ",value=selected_row["question"].iloc[0])
                        updA = st.text_area("Answer", key="updA",value=selected_row["answer"].iloc[0])
                        delB = st.form_submit_button(label="Delete")
                        updB = st.form_submit_button(label="Update")
                        question_source = persisted_qa_df[persisted_qa_df["question"] == sel]["source"].iloc[0]
                    if delB:
                        with st.spinner(""):
                            from langchain_core.documents import Document
                            full_list_except_selected = persisted_qa_df[persisted_qa_df["question"] != sel]                            
                            # rebuilt entries are admin-reviewed, so they are stamped with the current admission cycle
                            docs = [load_text_question_answer(d.question, d.source, d.answer) for d in full_list_except_selected.itertuples(index=False)]
                            vector_store = GlobalVariables.Cached_Question_answer_Vector_Store
                            record_manager = GlobalVariables.Cached_Question_answer_Record_Manager
                            upload_data(vector_store, record_manager, docs,cleanup="full")
                            st.success("Question Deleted Successfully")
                    
                    if updB:
                        with st.spinner(""):
                            ai_generated_document = load_text_question_answer(updQ,question_source, updA)
                            vector_store = GlobalVariables.Cached_Question_answer_Vector_Store
                            record_manager = GlobalVariables.Cached_Question_answer_Record_Manager
                            upload_data(vector_store, record_manager, [ai_generated_document])
                            st.success("Question Updated Successfully")

    




if __name__ == '__main__':
    main()
