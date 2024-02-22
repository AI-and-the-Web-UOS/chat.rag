from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.llms import HuggingFacePipeline
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.schema import format_document
from getpass import getpass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import pipeline
import os
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from mlx_lm import load, generate
from typing import Any, List, Mapping, Optional
import json
import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

path_db_files = "Database"
HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

class CustomLLM(LLM):
    n: int
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        result = generate(self.model, self.tokenizer, prompt=prompt, verbose=False, max_tokens=200)
        self.tokenizer.encode(result)
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            )
    chunks = text_splitter.split_documents(docs)
    return chunks

def load_model(type):
    # if type == "openai":
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=API_KEY_OPENAI)
    if type == "huggingface":
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device':'mps'}, 
            encode_kwargs={'normalize_embeddings': False},
        )
        #embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_KEY_HUGGINGFACE, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        #embeddings = HuggingFaceEmbeddings()
    return embeddings

def prep_vector_db(path_dir, model_type="huggingface"):
    print("Prepping vector DB")
    # load documents
    raw_documents = DirectoryLoader(path_dir, loader_cls=TextLoader, recursive=True, show_progress=True, use_multithreading=True, max_concurrency=8).load() # glob="**/*.txt",
    print(f"Found {len(raw_documents)} documents")
    print("Chunking documents")
    documents = get_text_chunks(raw_documents)
    print(f"Created {len(documents)} chunks")
    
    print("Embedding documents and adding to vector db.")
    embeddings = load_model(model_type)
    #text_embeddings = embeddings.embed_documents(documents)
    #text_embedding_pairs = zip(texts, text_embeddings)
    #text_embedding_pairs_list = list(text_embedding_pairs)
    #db = FAISS.from_embeddings(text_embedding_pairs_list, embeddings)
    db = FAISS.from_documents(documents, embeddings)
    print("Finished setting up vector db")
    return db

def _combine_documents(docs, document_prompt=PromptTemplate.from_template(template="{page_content}"), document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def answer_question(query, db, model_type="huggingface"):
    print("Attempting to answer question.")
    
    # ERROR: requests.exceptions.ReadTimeout: (ReadTimeoutError("HTTPSConnectionPool(host='api-inference.huggingface.co', port=443): Read timed out. (read timeout=120)"), '(Request ID: d22641da-0283-4140-8b9f-6f43c809c96a)')
    reply = rag_chain.invoke(query)

    return reply

@app.route('/query', methods=['POST'])
def query_handler():
    data = request.get_json()
    if not data or 'content' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    query = data['content']
    answers = answer_question(query, db)
    
    message_content = answers
    message_sender = "Harvey Specter"
    message_timestamp = datetime.datetime.now().isoformat()
    
    return jsonify({'content':message_content, 'sender':message_sender, 'timestamp':message_timestamp}), 200

if __name__ == "__main__":
    db = prep_vector_db(path_db_files)
    retriever = db.as_retriever()
    
    template = """Du bist der deutsche Harvey Specter, ein berühmter aber arroganter Anwalt mit einem Faible für gute Filmzitate.
        Du hast dich auf deutsches Recht spezialisiert und antwortest auf alle Fragen über deutsches Recht ehrlich und aufrichtig aber mit einem passiv-aggressiven Unterton.
        Wenn du dir bei einer Frage unsicher bist, gibst du keine falsche Antwort sondern findest einen frechen, passiv-aggressiven oder unverschämten Weg die Frage abzuweisen oder zu umgehen ohne Beleidigungen zu benutzen.
        Nutze zur Beantwortung der Fragen die Textausschnitte, die als Kontext gegeben sind.
        
        Kontext: {context}
        Frage: {query}
        """
    prompt = ChatPromptTemplate.from_template(template)
    
    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-8-bit-mlx")
    mlx_llm = CustomLLM(n=100, model=model, tokenizer=tokenizer)

    rag_chain = (
            {"context": retriever | _combine_documents, "query": RunnablePassthrough()}
            | prompt
            | mlx_llm
            | StrOutputParser()
    )
    app.run(port=5120, debug=True)
