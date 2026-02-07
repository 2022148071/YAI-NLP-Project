from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def store_article_to_chroma(article_text: str, collection_name: str = "articles"):
    """
    Store target article to Chroma vector database.
    
    Args:
        article_text: The article content to store
        collection_name: Name of the collection in Chroma
    """
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(article_text)
    
    # Create embeddings and store in Chroma
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    return vector_db

# Example usage
if __name__ == "__main__":
    sample_article = "Your article text here..."
    db = store_article_to_chroma(sample_article)
    print("Article stored successfully in Chroma!")