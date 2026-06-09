from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def rag_chatbot():
    load_dotenv()
    loader = DirectoryLoader(
        path="./papers",
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    
    # MARKDOWN_SEPARATORS là thứ tự tiêu chí để chia: độ ưu tiên là từ trên xuống nên chia theo tiêu đề, chia theo đoạn code,....
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    # cài đặt các thông số để chuẩn bị chia
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, #tổng số ký tự của đoạn chunk không được vượt quá 1200
        chunk_overlap=200, #chồng lấn ký tự
        add_start_index=True, #thêm số thứ tự cho đoạn chunk, là sô thứ tự của ký tự đầu tiền cả chunk so với toàn bộ file
        strip_whitespace=True, #xóa khoảng trắng ở 2 đầu chunk
        separators=MARKDOWN_SEPARATORS #dựa vào các tiêu chí để mà chia nhỏ
    )
    
    
    # chia và lưu vào biến splits
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )
    retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.2}
    )
    # Prompt
    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: \"I don't know based on the provided context.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template)
    # LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        user_input = input("Question: ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        answer = rag_chain.invoke(user_input)
        print(answer)

if __name__ == '__main__':
    rag_chatbot()






