import streamlit as st
from utilities._admin_utils import check_credentials,chroma_db_for_domain_rag
from utilities._globals import GLOBAL_VARIABLES as GlobalVariables
from dotenv import load_dotenv


st.set_page_config(page_title="Login App", layout="centered")
# Inject custom CSS to hide the sidebar



if "logged_in" not in st.session_state or not st.session_state['logged_in']:
    st.session_state.logged_in = False
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
    )
        
# Main Streamlit app
def main():     
    if not st.session_state.logged_in:
        # Login form
        st.title("Login")
        load_dotenv()
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")
        
        if login_button:
            with st.spinner("Processing..."):
                login_stat = check_credentials(username, password)
                if login_stat:
                    chroma_db_for_domain_rag()
                    GlobalVariables.initialize()
                    st.success("Login successful!.")
                    st.session_state.logged_in = True
                    st.markdown(
                        """
                        <style>
                        [data-testid="stSidebar"] {display: block;}
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    else:
        # Dashboard
        st.title("Welcome to FUE Admission Portal!")
        columns = st.columns(2)
        with columns[0]:            
            # st.write("Welcome to FUE Admission Portal!")
            logout_button = st.button("Logout", key="logout_button")
            
            if logout_button:
                st.session_state.logged_in = False
                st.rerun()

        st.image('fue_research_project_admin_part/bk.png')
       

if __name__ == "__main__":
    main()
