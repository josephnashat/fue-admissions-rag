from importlib import metadata
import streamlit as st
from langchain_core.messages import HumanMessage
from utilities.self_rag.graph import build_graph
from utilities._admin_utils import get_chat_channel,persist_client_chat, get_all_chat_channels, add_new_chat_channel,get_chat_channel_by_id,kyc_reached
import json
from markdown import markdown  
import pandas as pd

# Define a function with multi-line logic
def set_advanced_mode():
    st.session_state.update({"mode": "Advanced"})
    last_user_msg = [msg for msg in st.session_state['messages'] if msg["role"] == "user"][-1]
    lst_msg = st.session_state.messages[-1]
    # st.session_state['chat_input'] =  last_user_msg["content"]
    st.session_state['chat_input'] =  "Human asked: " + last_user_msg["content"] + "\n\n - AI Replied: " + lst_msg["content"] + "\n\n - Hmuan asked: I need more details" 
    
    
def is_arabic(text):
    import re
    # Regular expression to check for Arabic characters
    arabic_pattern = re.compile("[\u0600-\u06FF]")
    return bool(arabic_pattern.search(text))

# @st.cache_resource
# # Function for streaming TTS directly
# def text_to_speech_stream(text, lang="en"):
#     from gtts import gTTS
#     import io
#     tts = gTTS(text, lang=lang)
#     audio_stream = io.BytesIO()
#     tts.write_to_fp(audio_stream)
#     audio_stream.seek(0)  # Reset the pointer to the beginning of the stream
#     return audio_stream


def main():
    try:
        st.set_page_config( layout="wide")
        st.title("Admissions @ FUE")
        st.divider()
        
        
        if "compiled_app" not in st.session_state:
            app = build_graph()
            st.session_state.compiled_app = app
        else:
            app = st.session_state.compiled_app
            
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Get last chat session and if not exist any create a new one 
        if "thread_id" not in st.session_state or st.session_state.thread_id is None:
            st.session_state.thread_id = get_chat_channel()
            
        if "mode" not in st.session_state:
            st.session_state.mode= "Simple"
            
        if "chat_input" not in st.session_state:
            st.session_state.chat_input = ""
            
       
            
        with st.sidebar:
            st.header("Chat Controls")
            chat_channels_df = get_all_chat_channels()
            channel_to_id = pd.Series(chat_channels_df["thread_id"].values, index=chat_channels_df["channel_name"]).to_dict()
            selected_key_str = [key for key in channel_to_id.keys() if st.session_state.thread_id in key][0]
            selected_index = list(channel_to_id.keys()).index(selected_key_str)
            saved_chat_sessions_sb = st.selectbox("Saved Chat Sessions: ", list(channel_to_id.keys()), key="saved_chat_sessions", index=selected_index)
            saved_chat_sessions = channel_to_id[saved_chat_sessions_sb]
            slider_cols = st.columns(3)
            with slider_cols[0]:
                btn_save_chat = st.button("Save", key="btn_save_chat")
            with slider_cols[1]:
                btn_load_chat_session = st.button("Load", key="btn_load_chat_session")                   
            with slider_cols[2]:
                btn_new_chat = st.button("New", key="btn_new_chat")   

            if btn_save_chat:
                ls = json.dumps(st.session_state.messages)
                persist_client_chat(st.session_state.thread_id, ls)
                st.success("Chat Persisted")

            if btn_new_chat:
                ls = json.dumps(st.session_state.messages)
                persist_client_chat(st.session_state.thread_id, ls)
                st.session_state.thread_id  = add_new_chat_channel(None)
                st.session_state.messages = []
                st.success("New Chat Created")

            if btn_load_chat_session:
                saved_chat_sessions_msgs = get_chat_channel_by_id(saved_chat_sessions)
                if saved_chat_sessions_msgs:
                    st.session_state.messages = json.loads(saved_chat_sessions_msgs[0])
                    st.session_state.thread_id = saved_chat_sessions
                    st.success("Chat Loaded")    
                else:
                    st.error("Chat Not Found")  

        for idx, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message(message["role"], avatar="fue_research_project_admin_part/fue.png"):
                    if is_arabic(message["content"]):
                        # if st.button("🔊", key=f"play_ar_{idx}"):
                        #     st.audio(text_to_speech_stream(message["content"], lang="ar"))
                        st.write(f'<div style="text-align: right; direction: rtl;">{markdown(message["content"])}</div>',unsafe_allow_html=True,)                    
                        # st.markdown(message["content"])
                    else:
                        # if st.button("🔊", key=f"play_en_{idx}"):
                        #     st.audio(text_to_speech_stream(message["content"], lang="en"))
                        st.markdown(message["content"])                    
            else:
                with st.chat_message(message["role"], avatar="fue_research_project_admin_part/student.jpg"):
                    st.markdown(message["content"])

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        if prompt := st.chat_input("Ask me anything?") or st.session_state['chat_input']:
            final_prompt = st.session_state['chat_input'] if st.session_state['chat_input'] else prompt
            final_prompt_display = "I need more details" if st.session_state['chat_input'] else prompt
            st.session_state['chat_input'] = ""
            with st.spinner('Please wait...'):
                st.session_state.messages.append({"role": "user", "content": final_prompt_display})
                with st.chat_message("user", avatar="fue_research_project_admin_part/student.jpg"):
                    st.markdown(final_prompt_display)        
                with st.chat_message("assistant",avatar="fue_research_project_admin_part/fue.png"):                    
                        res = app.invoke({"messages": [HumanMessage(content=final_prompt, metadata={"mode": st.session_state.mode})]}, config=config)
                        if 'generation' in res:                                      
                            if is_arabic(res['generation']):
                                st.write(f'<div style="text-align: right; direction: rtl;">{markdown(res['generation'])}</div>',unsafe_allow_html=True,)                    
                                # st.markdown(message["content"])
                            else:
                                st.markdown(res['generation'])
                            # st.button("Dive Deeper...", key="btn_advanced", on_click=lambda: st.session_state.update({"mode": "Advanced"}))
                            st.session_state.messages.append({"role": "assistant", "content": res['generation']})
                            kyc_complete = kyc_reached(st.session_state.thread_id)
                            if st.session_state.mode == "Simple" and kyc_complete is not None:                                                    
                                st.button("Dive Deeper...", key="btn_advanced", on_click=set_advanced_mode ) 
                                                                                       
                            st.session_state.update({"mode": "Simple"})
    except Exception as e:
        # Catch all other exceptions
        st.error(f"An unexpected error occurred: {e}")
                


        

if __name__ == "__main__":
    main()




                        # if 'documents' in res:
                        #     images = [doc.metadata['images'] for doc in res["documents"] if 'images' in doc.metadata]                    
                        #     flattened_list = list(set([item for sublist in images for item in sublist]))
                        #     if flattened_list and len(flattened_list) > 0:
                        #         st.divider()   
                        #         num_columns = len(flattened_list)
                        #         columns = st.columns(num_columns)                         
                        #         for idx, image_path in enumerate(flattened_list):
                        #             # Select a column to display the image
                        #             with columns[idx % num_columns]:
                        #                 st.image(image_path, use_container_width=True)
                        #         st.divider()