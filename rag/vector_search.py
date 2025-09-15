from .web_search import WebSearcher

def get_relevant_chunks(query, vectordb, k=3):
    if vectordb is None:
        return []
    return vectordb.similarity_search(query, k=k)

def get_relevant_chunks_hybrid(query, vectordb, use_web_search=True, k=3, max_web_results=3):
    local_chunks = []
    web_results = []
    
    if vectordb is not None:
        local_chunks = vectordb.similarity_search(query, k=k)
    
    if use_web_search:
        searcher = WebSearcher()
        web_results = searcher.search_and_extract(query, max_web_results)
    
    return local_chunks, web_results

def get_relevant_chunks_with_context(query, vectordb, conversation_history=None, k=3):
    """NEW: Enhanced search considering conversation context"""
    if vectordb is None:
        return []
    
    enhanced_query = query
    
    if conversation_history:
        recent_questions = [conv["question"] for conv in conversation_history[-2:]]
        if recent_questions:
            context_keywords = " ".join(recent_questions)
            enhanced_query = f"{query} {context_keywords}"
    
    chunks = vectordb.similarity_search(enhanced_query, k=k)
    
    original_chunks = vectordb.similarity_search(query, k=k//2 if k > 2 else 1)
    
    all_chunks = chunks + original_chunks
    seen_content = set()
    unique_chunks = []
    
    for chunk in all_chunks:
        chunk_id = chunk.page_content[:100]
        if chunk_id not in seen_content:
            seen_content.add(chunk_id)
            unique_chunks.append(chunk)
            if len(unique_chunks) >= k:
                break
    
    return unique_chunks

def get_relevant_chunks_hybrid_with_memory(query, vectordb, conversation_history=None, 
                                         use_web_search=True, k=3, max_web_results=3):
    """NEW: Most advanced search function with memory and hybrid approach"""
    local_chunks = []
    web_results = []
    
    if vectordb is not None:
        local_chunks = get_relevant_chunks_with_context(
            query, vectordb, conversation_history, k=k
        )
    
    if use_web_search:
        enhanced_web_query = query
        if conversation_history:
            recent_context = conversation_history[-1]["question"] if conversation_history else ""
            if recent_context and recent_context != query:
                enhanced_web_query = f"{query} context: {recent_context}"
        
        searcher = WebSearcher()
        web_results = searcher.search_and_extract(enhanced_web_query, max_web_results)
    
    return local_chunks, web_results