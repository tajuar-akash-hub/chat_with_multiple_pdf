import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai


#load api keys 
load_dotenv()
os.getenv("GOOGLE_API_KEY")

#configure google api key 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




model = genai.GenerativeModel('gemini-1.5-flash')


# #get pdf function 
def get_pdf_to_text(pdfs):
    text = ""
    if pdfs:
        for pdf in pdfs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text+= page.extract_text()
                
    
    return text 
#get pdf function 
# def get_pdf_to_text(pdfs):
#     text = ""
#     if pdfs:
#         for pdf in pdfs:
#             with pdfplumber.open(pdf) as pdf_reader:
#                 for page in pdf_reader.pages:
#                     text += page.extract_text()
                
#     # st.write(text)
#     return text 


# get chunks of text 

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap = 1000)
    chunked_text = text_splitter.split_text(text)

    # # Print each chunk to see its content
    # for i, chunk in enumerate(chunked_text):
    #     print("chunk = --------",i)
    #     print(f"Chunk {i + 1}:")
    #     print(chunk)
    #     print("^^^^^^" * 50)  # Separator to distinguish between chunks

    return chunked_text

#Vector Store 
def get_vector_store(chunked_text):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vector_store = FAISS.from_texts(chunked_text,embedding = embedding)
    vector_store.save_local("faiss_index")
    # st.write("Saved done")
    # st.write("printing chunked text")
    # st.write(chunked_text[0])

# Conversational Client 


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
    # model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])

    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)
    
    return chain

def user_input(question):
    # Similary search 
    embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # query_embedding = faiss
    new_db = FAISS.load_local("faiss_index",embedding,allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(question)

    # st.write(docs)
    chain = get_conversational_chain()

    response = chain(
        {'input_documents':docs,
         'question' : question,
         } , 
         return_only_outputs=True ,
    )
    # print(response)

    st.write(response['output_text'])


def main():
    question = st.text_area("Ask question")
    button = st.button("Enter",key="Enter")
    if question:
        user_input(question)


    #markdown for button and text area -----------------------
    # Center the text area and button using CSS
    st.markdown("""
    <style>
        .stTextArea textarea {
            width: 500px;
            height: 200px;
            margin: 0 auto; /* Centers the text area */
            display: block;
        }
        .stButton button {
            width: 200px;
            height: 60px;
            font-size: 18px;
            margin: 0 auto; /* Centers the button */
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)
    #markdown for button and text area end -----------------------


    with st.sidebar:
        st.title("Sidebar")
        uploaded_pdfs = st.file_uploader("Upload your pdf",type="pdf", accept_multiple_files = True)
        if uploaded_pdfs:
            text = get_pdf_to_text(uploaded_pdfs)
            # st.write("reading the text")
            # st.write("here's the text")
            # st.write(text)
            chunked_text = get_text_chunks(text)
            # st.write(chunked_text)
            
            # st.write(chunked_text)
            if chunked_text:
                get_vector_store(chunked_text)
                # st.write("inside vector store")
            # st.write("done")
        # st.write(chunked_text)



        
if __name__ == "__main__":
    main()



# # print(response.text)


# user_question = st.text_input("Ask a Question from the PDF Files", height=200)
# centered_button = st.button("Calculate", key="calculate")

# if user_question:
#     response = model.generate_content(user_question)
#     st.write(response.text)

