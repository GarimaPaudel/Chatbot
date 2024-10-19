import time
import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
import dateparser

# Load environment variables
load_dotenv()

    # Load the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Streamlit title
st.title("RAG Application built on Gemini Model")

# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload a PDF file (Optional)", type="pdf")

# Initialize retriever as None
retriever = None

def validate_email(email):
    """Validate email address format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email)

def validate_phone(phone):
    """Validate phone number format (only digits, minimum 10 digits)."""
    return phone.isdigit() and len(phone) >= 10

def get_user_info():
    """Form to collect user's contact details."""
    with st.form(key="contact_form"):
        name = st.text_input("Enter your Name", key="name")
        phone = st.text_input("Enter your Phone Number", key="phone")
        email = st.text_input("Enter your Email", key="email")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if not name:
            st.error("Please enter your name.")
        elif not validate_phone(phone):
            st.error("Please enter a valid phone number with at least 10 digits.")
        elif not validate_email(email):
            st.error("Please enter a valid email address.")
        else:
            # If all fields are valid, process the info
            st.success(f"Thank you, {name}! We will contact you soon at {phone} or {email}.")
            # You can now store or process the information as needed
            # For example, you could send an email or save the data to a database.
            return {"name": name, "phone": phone, "email": email}

    return None

# Tool to book an appointment
def book_appointment_tool(name, date):
    print(f"Booking appointment for {name} on {date}")
    return f"Appointment successfully booked for {name} on {date}"

# Tool function definition
appointment_tool = Tool(
    name="book_appointment",
    func=book_appointment_tool,
    description="Tool for booking an appointment for the user."
)

# Initialize the agent with tools
tools = [appointment_tool]
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# Function to extract and validate date from the query
def extract_and_validate_date(query):
    date_str = re.search(r'\b(\d{4}-\d{2}-\d{2}|next \w+|tomorrow|today|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|in \d+ days)\b', query, re.IGNORECASE)
    
    if date_str:
        appointment_date = dateparser.parse(date_str.group(), settings={'PREFER_DATES_FROM': 'future'})
        
        if appointment_date is not None:
            formatted_date = appointment_date.strftime('%Y-%m-%d')
            return formatted_date
    return None

def handle_appointment(query):
    """Handle appointment booking requests."""
    appointment_date = extract_and_validate_date(query)
    if appointment_date is not None:
        user_info = get_user_info()  # Collect user information
        if user_info:
            response = book_appointment_tool(user_info['name'], appointment_date)
            st.success(response)
    else:
        st.error("No valid date found in the query. Please specify a valid date.")

# Function to handle call requests
def handle_call_me():
    user_info = get_user_info()
    if user_info:
        st.write(f"We will call you soon at {user_info['phone']}!")

# Function to process user queries related to the document
def handle_document_query(query, retriever):
    if retriever is None:
        st.error("No document uploaded to query from.")
        return

    # Define the system prompt for question answering
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "{context}"
    )

    # Define the prompt structure
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the retrieval and question-answering chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Retrieve relevant context from the document using the retriever
    retrieved_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Get the response from the document retrieval chain
    response = rag_chain.invoke({"input": query, "context": "{context}"})

    # Display the response
    st.write(response["answer"])

# Main function to route user queries
def process_user_query(query, retriever):
    if "call me" in query.lower():
        handle_call_me()
    
    elif "appointment" in query.lower():
        handle_appointment(query)
    
    else:
        handle_document_query(query, retriever)



if uploaded_file is not None:
    # Load and process the uploaded PDF
    with open("temp_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader("temp_pdf.pdf")
    data = loader.load()

    # Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# User input query for the chatbot
query = st.text_input("Ask a question :", key="input_question")

if query:
    process_user_query(query, retriever)
    