import streamlit as st
from rag_pipeline import *

st.set_page_config(page_title="PDF Analyzer", page_icon="🤖", layout="wide")

st.sidebar.title("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

st.title("🤖 PDF Analyzer Chatbot")
st.markdown("Ask questions from your uploaded document")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if uploaded_file:
    with st.spinner("Processing document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        docs = load_docs("temp.pdf")
        chunks = split_docs(docs)
        embeddings = create_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)
        llm = load_llm()

        st.session_state.qa_chain = create_qa_chain(vectorstore, llm)
        st.session_state.messages = []  # clear chat on new upload

    st.success("Document processed successfully!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain(question)

            answer = result.get("result", "No answer found.")
            sources = result.get("source_documents", [])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

            if sources:
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")

                        if "page" in doc.metadata:
                            st.write(f"Page: {doc.metadata['page']}")

                        st.write(doc.page_content[:300] + "...")
                        st.markdown("---")
