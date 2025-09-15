import streamlit as st
from rag.document_processing import load_and_split_documents, create_vectorstore
from rag.vector_search import get_relevant_chunks_hybrid
from rag.answer_generation import generate_answer_with_memory

def main():
    st.title("RAG Chatbot (Local + Web Search + Memory)")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.header("Beállítások")
        use_web_search = st.checkbox("Internetes keresés engedélyezése", value=True)
        search_mode = st.selectbox(
            "Keresési mód",
            ["hybrid", "local_only", "web_only"],
            index=0,
            help="Hybrid: helyi + web, Local only: csak helyi dokumentumok, Web only: csak internetes keresés"
        )
        max_web_results = st.slider("Max internetes eredmények", 1, 10, 3)
        
        st.header("Memória beállítások")
        use_memory = st.checkbox("Beszélgetés memória engedélyezése", value=True)
        max_memory_length = st.slider("Max memorizált üzenetek", 2, 20, 6)
        
        if st.button("Beszélgetés törlése"):
            st.session_state.conversation_history = []
            st.session_state.chat_messages = []
            st.rerun()

    if "vectordb" not in st.session_state:
        if search_mode != "web_only":
            with st.spinner("Helyi dokumentumok feldolgozása..."):
                try:
                    docs = load_and_split_documents()
                    if docs:
                        st.session_state.vectordb = create_vectorstore(docs)
                        st.success(f"{len(docs)} dokumentum betöltve!")
                    else:
                        st.warning("Nincsenek dokumentumok a 'documents' mappában")
                        st.session_state.vectordb = None
                except Exception as e:
                    st.error(f"Hiba a dokumentumok betöltésekor: {e}")
                    st.session_state.vectordb = None
        else:
            st.session_state.vectordb = None

    if st.session_state.chat_messages:
        st.markdown("### Beszélgetés előzmények")
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f"**Te:** {message['content']}")
            else:
                st.markdown(f"**Asszisztens:** {message['content']}")
        st.markdown("---")

    query = st.text_input("Kérdésed:", key="current_query")

    if query:
        st.session_state.chat_messages.append({"role": "user", "content": query})
        
        with st.spinner("Keresés és válaszgenerálás..."):
            try:
                local_chunks, web_results = get_relevant_chunks_hybrid(
                    query, 
                    st.session_state.vectordb if search_mode != "web_only" else None,
                    use_web_search=(use_web_search and search_mode != "local_only"),
                    max_web_results=max_web_results
                )
                
                conversation_context = []
                if use_memory and st.session_state.conversation_history:
                    conversation_context = st.session_state.conversation_history[-max_memory_length:]
                
                answer = generate_answer_with_memory(
                    query, 
                    local_chunks, 
                    web_results, 
                    conversation_context
                )

                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                
                st.session_state.conversation_history.append({
                    "question": query,
                    "answer": answer,
                    "local_chunks": len(local_chunks),
                    "web_results": len(web_results)
                })

                st.markdown("### Válasz")
                st.write(answer)

                with st.expander("Források megtekintése"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if local_chunks:
                            st.markdown("#### Helyi források")
                            for i, chunk in enumerate(local_chunks):
                                st.markdown(f"**Forrás {i+1}:** {chunk.metadata.get('source', 'ismeretlen')}")
                                st.text(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)
                                st.markdown("---")
                    
                    with col2:
                        if web_results:
                            st.markdown("#### Internetes források")
                            for i, result in enumerate(web_results):
                                st.markdown(f"**Web {i+1}:** [{result['title'][:50]}...]({result['url']})")
                                st.text(f"{result['content'][:200]}..." if len(result['content']) > 200 else result['content'])
                                st.markdown("---")

                if use_memory:
                    st.markdown(f"*Memóriában: {len(st.session_state.conversation_history)} interakció*")

            except Exception as e:
                st.error(f"Hiba történt: {e}")

if __name__ == "__main__":
    main()