from langchain_community.llms import Ollama
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate

def generate_answer(query, chunks):
    """Original function for backward compatibility - KEEP THIS"""
    llm = Ollama(model="mistral")
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    answer = chain.run({"input_documents": chunks, "question": query})
    return answer

def generate_answer_enhanced(query, local_chunks, web_results):
    """Enhanced answer generation using both local and web sources - EXISTING function"""
    llm = Ollama(model="mistral")
    
    local_context = ""
    if local_chunks:
        local_context = "\n\n--- HELYI DOKUMENTUMOK ---\n"
        for i, chunk in enumerate(local_chunks):
            local_context += f"\nForrás {i+1} ({chunk.metadata.get('source', 'ismeretlen')}):\n{chunk.page_content}\n"
    
    web_context = ""
    if web_results:
        web_context = "\n\n--- INTERNETES FORRÁSOK ---\n"
        for i, result in enumerate(web_results):
            web_context += f"\nWeb forrás {i+1} ({result['url']}):\n{result['content']}\n"
    
    prompt_template = PromptTemplate(
        input_variables=["query", "local_context", "web_context"],
        template="""
        Válaszolj a kérdésre a rendelkezésre álló információk alapján. 
        Használd mind a helyi dokumentumokat, mind az internetes forrásokat.
        Ha valamely forrásból nincs releváns információ, jelezd ezt.
        Hivatkozz a forrásokra a válaszodban.
        
        {local_context}
        {web_context}
        
        Kérdés: {query}
        
        Átfogó válasz:
        """
    )
    
    formatted_prompt = prompt_template.format(
        query=query,
        local_context=local_context,
        web_context=web_context
    )
    
    return llm(formatted_prompt)

def generate_answer_with_memory(query, local_chunks, web_results, conversation_history):
    """NEW: Enhanced answer generation with conversation memory"""
    llm = Ollama(model="mistral")
    
    conversation_context = ""
    if conversation_history:
        conversation_context = "\n\n--- KORÁBBI BESZÉLGETÉS ---\n"
        for i, conv in enumerate(conversation_history):
            conversation_context += f"\nKorábbi kérdés {i+1}: {conv['question']}\n"
            conversation_context += f"Korábbi válasz {i+1}: {conv['answer'][:300]}...\n"
    
    local_context = ""
    if local_chunks:
        local_context = "\n\n--- HELYI DOKUMENTUMOK ---\n"
        for i, chunk in enumerate(local_chunks):
            local_context += f"\nForrás {i+1} ({chunk.metadata.get('source', 'ismeretlen')}):\n{chunk.page_content}\n"
    

    web_context = ""
    if web_results:
        web_context = "\n\n--- INTERNETES FORRÁSOK ---\n"
        for i, result in enumerate(web_results):
            web_context += f"\nWeb forrás {i+1} ({result['url']}):\n{result['content']}\n"
    
    prompt_template = PromptTemplate(
        input_variables=["query", "conversation_context", "local_context", "web_context"],
        template="""
        Te egy intelligens asszisztens vagy, aki válaszol a kérdésekre a rendelkezésre álló információk alapján.
        
        FONTOS UTASÍTÁSOK:
        1. Használd a korábbi beszélgetést a kontextus megértéséhez
        2. Ha a jelenlegi kérdés kapcsolódik korábbi kérdésekhez, hivatkozz rájuk
        3. Használd mind a helyi dokumentumokat, mind az internetes forrásokat
        4. Ha nincs releváns információ, jelezd ezt
        5. Legyél következetes a korábbi válaszaiddal
        6. Ha ellentmondás van a források között, jelezd ezt
        
        {conversation_context}
        {local_context}
        {web_context}
        
        Jelenlegi kérdés: {query}
        
        Átfogó válasz (figyelembe véve a korábbi beszélgetést):
        """
    )
    
    formatted_prompt = prompt_template.format(
        query=query,
        conversation_context=conversation_context,
        local_context=local_context,
        web_context=web_context
    )
    
    return llm(formatted_prompt)

def format_conversation_history(conversation_history, max_length=3):
    """Helper function to format conversation history for display"""
    if not conversation_history:
        return "Nincs korábbi beszélgetés."
    
    formatted = "Korábbi beszélgetés:\n"
    recent_history = conversation_history[-max_length:]
    
    for i, conv in enumerate(recent_history, 1):
        formatted += f"\n{i}. Kérdés: {conv['question']}\n"
        formatted += f"   Válasz: {conv['answer'][:150]}...\n"
    
    return formatted