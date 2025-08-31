import sqlite3
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Optional: colored console output
class bcolors:
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

# -------------------------------
# 1️⃣ Structured output schema
# -------------------------------
class MaterialAnalysis(BaseModel):
    material: str = Field(description="Name of the material")
    max_temperature: int = Field(description="Maximum operating temperature in Celsius")
    suitability: str = Field(description="Yes/No/Borderline")
    reasoning: str = Field(description="Step-by-step justification")

parser = PydanticOutputParser(pydantic_object=MaterialAnalysis)
format_instructions = parser.get_format_instructions()

# -------------------------------
# 2️⃣ SQLite database setup
# -------------------------------
conn = sqlite3.connect("materials.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS material_temps (
    material TEXT PRIMARY KEY,
    max_temperature INTEGER
)
""")

materials_data = [
    ("Aluminum 7075", 200),
    ("Inconel 718", 700),
    ("Ti-6Al-4V", 500)
]
cursor.executemany("INSERT OR REPLACE INTO material_temps VALUES (?, ?)", materials_data)
conn.commit()

# -------------------------------
# 3️⃣ Retriever Tool: extracts only material name
# -------------------------------
def sqlite_retriever(tool_input: str) -> str:
    """
    Extracts material name from tool input and fetches max temperature from DB.
    """
    # If input contains "Material: ..." extract it
    if "Material:" in tool_input:
        material_name = tool_input.split("Material:")[1].split("\n")[0].strip()
    else:
        material_name = tool_input.strip()

    cursor.execute(
        "SELECT max_temperature FROM material_temps WHERE LOWER(material)=LOWER(?)",
        (material_name,)
    )
    result = cursor.fetchone()
    return str(result[0]) if result else "unknown"

retriever_tool = Tool(
    name="MaterialRetriever",
    func=sqlite_retriever,
    description="Fetch the maximum operating temperature for a given material from SQLite DB (case-insensitive)."
)

# -------------------------------
# 4️⃣ LLM for reasoning agent
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

reasoning_prompt = ChatPromptTemplate.from_template(
    """
You are a materials analysis assistant.
Determine if a material is suitable for a required operating temperature.

Material: {material}
Required max operating temp: {temp_req}C

Use the tool 'MaterialRetriever' to fetch the material's max temperature.
Provide structured JSON output with reasoning.
{format_instructions}
"""
)

# -------------------------------
# 5️⃣ Initialize agent
# -------------------------------
agent = initialize_agent(
    tools=[retriever_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------------------------------
# 6️⃣ Run agentic workflow
# -------------------------------
material_input = "Aluminum 7075"
temp_req_input = 250

agent_input = f"Material: {material_input}\nRequired temperature: {temp_req_input}C\n{format_instructions}"
result_text = agent.run(agent_input)

print(f"{bcolors.OKGREEN}Agent Response:{bcolors.ENDC}")
print(result_text)

# -------------------------------
# 7️⃣ Close DB
# -------------------------------
conn.close()