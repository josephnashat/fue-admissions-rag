import streamlit as st
from IPython.display import Image
from utilities.self_rag.graph import build_graph
from utilities._globals import GLOBAL_VARIABLES as GlobalVariables

def main():
    st.set_page_config( layout="centered")
    
    st.title("Main System Architecture")

    st.divider()
    if "compiled_app" not in st.session_state:
        app = build_graph()
        st.session_state.compiled_app = app
    else: 
        app = st.session_state.compiled_app
    # st.image('fue_research_project_admin_part/graph.png')
    st.image(Image(app.get_graph(xray=True).draw_mermaid_png()).data)
    


if __name__ == "__main__":
    main()
    
    
    
    
        # cols = st.columns(2)
    # with cols[0]:
    #     modes = ["Simple", "Advanced"]
    #     currentmode = GlobalVariables.Current_Mode
    #     selected_mode = st.selectbox("Operation Mode:", options=modes, index=modes.index(currentmode))
    # with cols[1]:
    #     btn_change_mode = st.button("Change Mode", key="btn_change_mode")
    #     if btn_change_mode:
    #         GlobalVariables.Current_Mode = selected_mode