import streamlit as st
from utilities._admin_utils import queries_list, load_chat_profiles, persist_applicants, rank_applicants_chat, load_perssisted_chat_messages, predict_applicants
import pandas as pd
import os

def main():
    st.set_page_config(page_title="Managing Chat History...", layout="wide")
    st.title("Managing Chat...💬 ")
    
    # Create tabs
    tab_titles = ['Manage Users'] #, 'Manage Chats']
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        st.header('Manage Users Initiated Chat')
        st.write('List of all users who started chat sessions')
        btn_load_applicants_data = st.button("Load Data", key="btn_load_applicants_data")
        if btn_load_applicants_data or 'btn_persist_and_rank' in st.session_state:
            st.session_state.btn_persist_and_rank = True
            with st.spinner('Wait for it...'):
                chat_profiles = [_profile.applicant_profile for _profile in load_chat_profiles(queries_list['only_kyc'])]
                df = pd.DataFrame(chat_profiles)
                unique_df = df.drop_duplicates(keep="last")
                st.dataframe(unique_df)
            btn_persist_rank = st.button("Persist, Rank, and Predict applicants", key="btn_persist_rank")
            if btn_persist_rank:                
                with st.spinner('Wait for it...'):  
                    ranked_df = rank_applicants_chat(unique_df)
                    file_exists = os.path.exists('fue_research_project_admin_part/modelsvm.pk1')
                    if not file_exists:
                        st.error('Model not found, prediction will not apply')
                        ranked_df = predict_applicants(ranked_df, 0)
                    else:
                        ranked_df = predict_applicants(ranked_df, 1)
                    persist_applicants(ranked_df)
            persisted_df = load_perssisted_chat_messages()
            st.dataframe(persisted_df)
            if 'user_id' in unique_df and len(unique_df) > 0:
                selected_id = st.selectbox("Show details for user_id: ", unique_df["user_id"])
                if selected_id:
                    with st.spinner('Wait for it...'):  
                        chat_profiles_main = load_chat_profiles(queries_list['all_user_meta_data'], selected_id)              
                        chat_profiles = [{'step':_profile.step,'type': _profile.key_step,'question': str(_profile.question), 'generation': _profile.generation} for _profile in chat_profiles_main if _profile.question != None]
                        df = pd.DataFrame(chat_profiles)
                        unique_df = df.drop_duplicates(keep="last")
                    st.dataframe(unique_df)  
                        
                    selected_id = st.selectbox("Show documents for generate transaction: ", [_profile.step for _profile in chat_profiles_main if _profile.document_sources and len(_profile.document_sources)>0])
                    if selected_id:
                        with st.spinner('Wait for it...'):                
                            chat_profiles = [zip(_profile.document_sources,  _profile.document_page_contents) for _profile in chat_profiles_main if _profile.step == selected_id ]                    
                            df = pd.DataFrame(list(chat_profiles[0]), columns=['document_sources', 'document_page_contents'])
                            unique_df = df.drop_duplicates(keep="last")
                        st.dataframe(unique_df)                
            

            




if __name__ == '__main__':
    main()
