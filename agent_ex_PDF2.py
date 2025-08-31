from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# -------------------------------
# 1️⃣ Define structured output schema
# -------------------------------
class MaterialAnalysis(BaseModel):
    material: str = Field(description="Name of the material")
    max_temperature: int = Field(description="Maximum operating temperature in Celsius")
    suitability: str = Field(description="Whether the material is suitable for the required temperature")
    reasoning: str = Field(description="Explanation for the suitability")

# Create parser
parser = PydanticOutputParser(pydantic_object=MaterialAnalysis)
format_instructions = parser.get_format_instructions()

# -------------------------------
# 2️⃣ Load PDF and create vector store
# -------------------------------
pdf_loader = PyPDFLoader("materials_table.pdf")
docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# -------------------------------
# 3️⃣ Define retriever tool
# -------------------------------
def pdf_retriever(query: str) -> str:
    """Fetch material info from the PDF vector store."""
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "unknown"
    return " ".join([r.page_content for r in results])

# -------------------------------
# 4️⃣ Define LLM
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# -------------------------------
# 5️⃣ Create LangGraph React Agent
# -------------------------------
agent = create_react_agent(
    model=llm,
    tools=[pdf_retriever],
    prompt=f"You are a helpful assistant. Provide output in this JSON schema:\n{format_instructions}"
)

# -------------------------------
# 6️⃣ Run the agent
# -------------------------------
material_input = "Aluminum 7075"
temp_req_input = 250

input_text = f"""
Material: {material_input}
Required temperature: {temp_req_input}C
"""

agent_response = agent.invoke({"messages": [{"role": "user", "content": input_text}]})

# -------------------------------
# 7️⃣ Extract AIMessage containing JSON
# -------------------------------
ai_text = None
for msg in agent_response['messages']:
    if msg.__class__.__name__ == "AIMessage" and msg.content.strip():
        ai_text = msg.content
        break

if not ai_text:
    raise RuntimeError("No AIMessage with content found.")

# Strip Markdown/code fences
ai_text_clean = ai_text.strip()
if ai_text_clean.startswith("```json"):
    ai_text_clean = ai_text_clean.replace("```json", "").replace("```", "").strip()

# -------------------------------
# 8️⃣ Parse JSON with PydanticOutputParser
# -------------------------------
try:
    parsed_result = parser.parse(ai_text_clean)
except Exception as e:
    print("Failed to parse AI output:")
    print(ai_text_clean)
    raise e

# -------------------------------
# 9️⃣ Print clean output
# -------------------------------
print(f"{bcolors.OKGREEN}Material: {bcolors.ENDC}{parsed_result.material}")
print(f"{bcolors.OKGREEN}Max Temp: {bcolors.ENDC}{parsed_result.max_temperature}°C")
print(f"{bcolors.OKGREEN}Suitable? {bcolors.ENDC}{parsed_result.suitability}")
print(f"{bcolors.OKGREEN}Reasoning: {bcolors.ENDC}{parsed_result.reasoning}")
