import streamlit as st
from rag import process_urls, generate_answer, create_qa_chain

st.set_page_config(

    page_title="Real Estate Research Tool",
    page_icon="🏡",
    layout="wide"
)

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🏡 Real Estate Research Tool")
st.subheader("Extract useful insights from the provided URLs.")

with st.expander("📌 Supported Websites"):
    st.markdown("""
This application is designed for websites that provide publicly accessible HTML content.

**Recommended:**
- CNBC
- Reuters
- BBC News
- Wikipedia
- Python.org
- Hugging Face Docs
- LangChain Docs
- Scikit-learn Docs
- PyTorch Docs
- TensorFlow Docs

**May Not Work:**
- 99acres
- MagicBricks
- Housing.com
- NoBroker
- Amazon
- Flipkart

These websites often use JavaScript rendering or anti-bot protection, which can prevent automatic content extraction.
""")

st.sidebar.header("Article URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("Process URLs", use_container_width=True)

status_placeholder = st.empty()


if process_url_button:

    urls = [url.strip() for url in (url1, url2, url3) if url.strip()]

    if not urls:
        status_placeholder.warning("⚠️ Please provide at least one URL.")

    else:

        # Process the URLs
        for status in process_urls(urls):
            status_placeholder.info(status)

        # Create a fresh retrieval chain
        st.session_state.retrieval_chain = create_qa_chain()

        # Mark this session as processed
        st.session_state.documents_processed = True

        # Clear previous chat history because documents changed
        st.session_state.chat_history = []

        status_placeholder.success("✅ URLs processed successfully!")

st.markdown("---")
st.subheader("Ask a Question")

query = st.text_input(
    "What's on your mind? Please type in your question."
)

if query.strip():

    if not st.session_state.documents_processed:
        st.warning("⚠️ Please process the URL(s) first.")

    else:
        try:

            answer, sources = generate_answer(
                query=query,
                retrieval_chain=st.session_state.retrieval_chain
            )

            # Save conversation
            st.session_state.chat_history.append({
                    "role": "user",
                    "content": query
                })

            st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            st.header("Answer")
            st.write(answer)

            if sources:
                st.markdown("### Sources")

                for source in sources:
                    st.markdown(f"- {source}")

        except Exception as e:
            st.error(f"Error : {e}")

else:
    st.info("💡 Please enter your question for which you want information about.")
