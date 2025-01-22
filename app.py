import streamlit as st
import os
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults


load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))


tool = TavilySearchResults(
    max_results=100,
    search_depth='advanced'
)

evaluation_prompt = ChatPromptTemplate.from_template("""
Analyze the provided resume and generate a concise evaluation within 10 lines. 
Rate the resume out of 10 based on the following criteria:
- Content relevance
- Clarity
- Structure
- Achievements
- Presentation
- Skills

Ensure the response includes:
1. **Overall Rating**: Start with the score out of 10 and provide a brief justification.
2. **Strengths**: Highlight positive aspects in 2-3 lines.
3. **Suggestions for Improvement**: Mention areas for enhancement in 2-3 lines.
4. **Suggest Skills**: Mention what skills the user is lacking.
5. **Job Roles**: Suggest relevant jobs.

Context:
{context}

Question: Evaluate this resume thoroughly and respond concisely in 10 lines.
""")

job_search_prompt = ChatPromptTemplate.from_template("""
Generate a concise search query based on the provided resume to find job opportunities.

Include:
1. **Search Query**: Generate a query tailored for job portals (e.g., Naukri, LinkedIn) and freelancing platforms (e.g., Upwork, Freelancer) but don't search for blogs and other learning platforms, search only for jobs.
2. **Relevant Roles**: Suggest roles that align with the skills and experience in the resume.

Context:
{context}

Question: Generate a search query for relevant job roles on specified platforms without any additional details or explanation and don't provide blogs and resources.
give only job opportunites related to specific role.provide only those urls where url must contain jobs keyword
""")

technical_prompt = ChatPromptTemplate.from_template("""
Based on the provided resume, generate 10 technical interview questions. These should cover:
1. Technical skills
2. Relevant projects
3. Industry-specific knowledge

Context:
{context}

Question: Generate 10 technical interview questions based on the resume.
""")

non_technical_prompt = ChatPromptTemplate.from_template("""
Based on the provided resume, generate 10 non-technical interview questions. These should focus on:
1. Behavioral aspects
2. Teamwork and collaboration
3. Problem-solving and leadership

Context:
{context}

Question: Generate 10 non-technical interview questions based on the resume.
""")

def process_resume(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    os.remove(temp_file_path)
    return vectorstore

st.title("Resume Toolkit")
st.sidebar.header("Upload Resume")
uploaded_file = st.sidebar.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    option = st.sidebar.radio("Choose an Option:", ["Resume Evaluation", "Job Search with Career Guidance", "Interview Preparation"])

    try:
        with st.spinner("Processing your request..."):
            vectorstore = process_resume(uploaded_file)
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents("Evaluate this resume thoroughly.")
            if option == "Resume Evaluation":
                qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=evaluation_prompt)
                evaluation_response = qa_chain.run(input_documents=docs, question="Evaluate this resume thoroughly.")
                st.success("Resume Evaluation Completed!")
                st.markdown(f"### Evaluation Report:\n\n{evaluation_response}")
            elif option == "Job Search with Career Guidance":
                qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=job_search_prompt)
                job_search_response = qa_chain.run(input_documents=docs, question="Generate a search query for job portals for relavent job roles.")
                search_results = tool.invoke({'query': f"{job_search_response} and search for all job portals"})
                st.success("Job Search Completed!")
                st.markdown("### Job Search Results:")
                if isinstance(search_results, list):  
                    for result in search_results:
                        st.write(f"- {result}")
                else:
                    st.write(search_results)

            elif option == "Interview Preparation":
                st.markdown("### Choose an Interview Type")
                interview_type = st.radio("Select Interview Type:", ["Technical Interview", "Non-Technical Interview"])

                if interview_type == "Technical Interview":
                    if st.button("Generate Technical Interview Questions"):
                        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=technical_prompt)
                        technical_questions = qa_chain.run(input_documents=docs, question="Generate 10 technical interview questions.")
                        st.success("Technical Interview Questions Generated!")
                        st.markdown("### Technical Interview Questions:")
                        st.markdown(technical_questions)

                elif interview_type == "Non-Technical Interview":
                    if st.button("Generate Non-Technical Interview Questions"):
                        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=non_technical_prompt)
                        non_technical_questions = qa_chain.run(input_documents=docs, question="Generate 10 non-technical interview questions.")
                        st.success("Non-Technical Interview Questions Generated!")
                        st.markdown("### Non-Technical Interview Questions:")
                        st.markdown(non_technical_questions)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a resume to begin.")

