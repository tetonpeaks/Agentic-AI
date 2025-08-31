from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# -------------------------------
# 1️⃣ Schema
# -------------------------------
class MaterialAnalysis(BaseModel):
    material: str = Field(description="Name of the material")
    max_temperature: int = Field(description="Maximum operating temperature in Celsius")
    suitability: str = Field(description="Whether the material is suitable for the required temperature")
    reasoning: str = Field(description="Explanation for the suitability")

parser = PydanticOutputParser(pydantic_object=MaterialAnalysis)
format_instructions = parser.get_format_instructions()

# -------------------------------
# 2️⃣ Load PDF
# -------------------------------
pdf_loader = PyPDFLoader("materials_table.pdf")  # your PDF file
docs = pdf_loader.load()

# -------------------------------
# 3️⃣ Split text for retrieval
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# -------------------------------
# 4️⃣ Create vector store using Chroma
# -------------------------------
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# -------------------------------
# 5️⃣ Retriever Tool
# -------------------------------
def pdf_retriever(query: str) -> str:
    results = vectorstore.similarity_search(query, k=2)  # top 2 relevant chunks
    if not results:
        return "unknown"
    return " ".join([r.page_content for r in results])

retriever_tool = Tool(
    name="PDFRetriever",
    func=pdf_retriever,
    description="Fetch material information from the PDF datasheet."
)

# -------------------------------
# 6️⃣ LLM + Agent
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

agent = initialize_agent(
    tools=[retriever_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------------------------------
# 7️⃣ Run
# -------------------------------
material_input = "Aluminum 7075"
temp_req_input = 250

agent_input = f"""
Material: {material_input}
Required temperature: {temp_req_input}C

{format_instructions}
"""

result_text = agent.invoke(agent_input)
print(result_text)
