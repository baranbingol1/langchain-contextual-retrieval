from rag_search import RAGSearchEngine, SearchConfig
from dotenv import load_dotenv, find_dotenv

# you can change the components to your preferred ones
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# make sure you have API keys set for preferred components in a .env file 
# and additional langchain dependencies installed
load_dotenv(find_dotenv(), override=True)

contextualizer = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)

search_engine = RAGSearchEngine(
    contextualizer=contextualizer,
    embedding_model=embeddings,
    vector_store=InMemoryVectorStore,
    # you can also modify the hyperparameters here
    config=SearchConfig(
        chunk_size=512,
        chunk_overlap=256,
        semantic_weight=0.7,
        lexical_weight=0.3,
        initial_k = 20,
        final_k = 5
    )
)

# sample documents
documents = [
    """Climate change is a long-term shift in global weather patterns and temperatures. 
    It is primarily caused by human activities that release greenhouse gases into the atmosphere.
    The main greenhouse gases include carbon dioxide from burning fossil fuels, methane from agriculture
    and livestock, and chlorofluorocarbons from industrial processes. These gases trap heat in the
    atmosphere, leading to global warming, rising sea levels, and extreme weather events. The impacts
    of climate change include melting glaciers, ocean acidification, and threats to biodiversity.""",
    
    """Renewable energy sources are sustainable alternatives to fossil fuels that help combat climate change.
    Solar power harnesses the sun's energy through photovoltaic cells and solar thermal collectors.
    Wind energy is captured by turbines that convert kinetic energy into electricity. Hydroelectric
    power generates electricity by using flowing water to turn turbines. Geothermal energy taps into
    Earth's internal heat. These clean energy sources reduce greenhouse gas emissions and provide
    energy security while creating new jobs in the green economy."""
]

# process documents
search_engine.process_documents(documents)

# perform search
query = "What are the relationships between greenhouse gas emissions, their sources, and their effects on global climate systems?"
results = search_engine.search(query)

print("Search Results:")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(result)
