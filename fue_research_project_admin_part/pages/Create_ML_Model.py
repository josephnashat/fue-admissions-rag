import streamlit as st
from utilities._admin_utils import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib


if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] =''
if 'cleaned_data_label' not in st.session_state:
    st.session_state['cleaned_data_label'] =''    
if 'sentences_train' not in st.session_state:
    st.session_state['sentences_train'] =''
if 'sentences_test' not in st.session_state:
    st.session_state['sentences_test'] =''
if 'labels_train' not in st.session_state:
    st.session_state['labels_train'] =''
if 'labels_test' not in st.session_state:
    st.session_state['labels_test'] =''
if 'svm_classifier' not in st.session_state:
    st.session_state['svm_classifier'] =''

def main():
    st.set_page_config( layout="wide")
    st.title("Building SVC Model Based on Collected Data...")
    
    # Create tabs
    tab_titles = ['Data Preprocessing', 'Model Training', 'Model Evaluation',"Save Model", "Load and Test Model"]
    tabs = st.tabs(tab_titles)

    # Adding content to each tab

    #Data Preprocessing TAB...
    with tabs[0]:
        st.header('Data Preprocessing')
        st.write('Process the data...')

        # Capture the CSV file
        data = st.file_uploader("Upload CSV file",type="csv")

        button = st.button("Load data",key="data")

        if button:
            with st.spinner('Wait for it...'):
                our_data=read_data(data)
                embeddings=get_embeddings()
                st.session_state['cleaned_data'], st.session_state['cleaned_data_label'] = create_embeddings(our_data,embeddings)
            st.success('Done!')


    #Model Training TAB
    with tabs[1]:
        st.header('Model Training')
        st.write('Train the model...')
        button = st.button("Train model",key="model")   
            
        if button:
            if 'cleaned_data' in st.session_state and st.session_state['cleaned_data']:
                with st.spinner('Wait for it...'):
                    st.session_state['sentences_train'], st.session_state['sentences_test'], st.session_state['labels_train'], st.session_state['labels_test']=split_train_test__data(st.session_state['cleaned_data'], st.session_state['cleaned_data_label'])
                    
                    st.session_state['svm_classifier']  = make_pipeline(StandardScaler(), SVC(class_weight='balanced')) 

                    # fit the support vector machine
                    st.session_state['svm_classifier'].fit(st.session_state['sentences_train'], st.session_state['labels_train'])
                st.success('Done!')
            else:
                st.error('Load model first')

    #Model Evaluation TAB
    with tabs[2]:
        st.header('Model Evaluation')
        st.write('Evaluate the model...')
        button = st.button("Evaluate model",key="Evaluation")

        if button:
            if 'sentences_train' in st.session_state and st.session_state['sentences_train']:
                with st.spinner('Wait for it...'):
                    accuracy_score=get_score(st.session_state['svm_classifier'],st.session_state['sentences_test'],st.session_state['labels_test'])
                    st.success(f"Validation accuracy is {100*accuracy_score}%!")
                    st.write("A sample run:")

                    text="""How competitive is admission to the pharmacy program? What documents are required to apply for the dentistry program?
                    """
                    score = 5
                    st.write("***Student Questions*** : "+text + "***Score***: " + str(score))

                    #Converting out TEXT to NUMERICAL representaion
                    embeddings= get_embeddings()
                    query_result = embeddings.embed_query(text)
                    import numpy as np
                    query_result = np.hstack([query_result, str(score)])
                    #Sample prediction using our trained model
                    result= st.session_state['svm_classifier'].predict([query_result])
                    st.write("***Expected to*** : "+result[0])
                st.success('Done!')
            else:
                st.error('Train model first')

    #Save model TAB
    with tabs[3]:
        st.header('Save model')
        st.write('Here we save the model...')

        button = st.button("Save model",key="save")
        if button:
            if st.session_state['svm_classifier']:
                with st.spinner('Wait for it...'):
                    joblib.dump(st.session_state['svm_classifier'], 'fue_research_project_admin_part/modelsvm.pk1')
                st.success('Done!')
            else:
                st.error('Evaluate model first')

    #Load model TAB
    with tabs[4]:
        st.header('Load model')
        st.write('Here we load the model to manually test it...')
        
        question = st.text_input("Enter the questions", key="question", value="How competitive is admission to the pharmacy program? What documents are required to apply for the dentistry program?")
        btn_evaluate_predict = st.button('Evaluate and Predict')
        if btn_evaluate_predict:
            #Converting out TEXT to NUMERICAL representaion
            file_exists = os.path.exists('fue_research_project_admin_part/modelsvm.pk1')
            if file_exists:        
                with st.spinner('Wait for it...'):
                    embeddings= get_embeddings()
                    query_result = embeddings.embed_query(question)
                    score = calculate_chat_rank(question)
                    import numpy as np
                    query_result = np.hstack([query_result, str(score)])
                    import joblib
                    Fitmodel = joblib.load('fue_research_project_admin_part/modelsvm.pk1')
                    result=Fitmodel.predict([query_result])
            
                st.write("***Score*** : "+score)
                st.write("***Expected to*** : "+result[0])
            else: 
                st.error('Save model first')

if __name__ == '__main__':
    main()