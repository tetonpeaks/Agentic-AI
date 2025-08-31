import sqlite3
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Optional: colored console output
class bcolors:
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

# -------------------------------
# 1️⃣ Define structured output schema
# -------------------------------
class MaterialAnalysis(BaseModel):
    material: str = Field(description="Name of the material")
    max_temperature: int = Field(description="Maximum operating temperature in Celsius")
    suitability: str = Field(description="Yes/No/Borderline")
    reasoning: str = Field(description="Step-by-step justification")

parser = PydanticOutputParser(pydantic_object=MaterialAnalysis)
format_instructions = parser.get_format_instructions()

# -------------------------------
# 2️⃣ Set up SQLite database (materials table)
# -------------------------------
conn = sqlite3.connect("materials.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS material_temps (
    material TEXT PRIMARY KEY,
    max_temperature INTEGER
)
""")

# Populate table with example data
materials_data = [
    ("Aluminum 7075", 200),
    ("Inconel 718", 700),
    ("Ti-6Al-4V", 500)
]

cursor.executemany("INSERT OR REPLACE INTO material_temps VALUES (?, ?)", materials_data)
conn.commit()

# -------------------------------
# 3️⃣ Retriever Agent (queries SQLite)
# -------------------------------
def retriever_agent(material_name: str):
    cursor.execute("SELECT max_temperature FROM material_temps WHERE material=?", (material_name,))
    result = cursor.fetchone()
    return result[0] if result else "unknown"

# -------------------------------
# 4️⃣ Reasoning Agent (LLM)
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

reasoning_prompt = ChatPromptTemplate.from_template(
    """
    You are a materials analysis assistant.
    Material: {material}
    Required max operating temp: {temp_req}C
    Retrieved max temperature: {retrieved_temp}C

    Use step-by-step reasoning to evaluate suitability.
    {format_instructions}
    """
)

# -------------------------------
# 5️⃣ Run multi-agent pipeline
# -------------------------------
material_input = "Aluminum 7075"
temp_req_input = 250

# Step 1: Retrieve from database
retrieved_temp = retriever_agent(material_input)

# Step 2: Reasoning + structured output
chain_input = {
    "material": material_input,
    "temp_req": temp_req_input,
    "retrieved_temp": retrieved_temp,
    "format_instructions": format_instructions
}

chain = reasoning_prompt | llm | parser
result = chain.invoke(chain_input)

# -------------------------------
# 6️⃣ Display results
# -------------------------------
print(f"{bcolors.OKGREEN}Result:{bcolors.ENDC}")
print(result.model_dump_json(indent=2))

# -------------------------------
# 7️⃣ Close DB connection
# -------------------------------
conn.close()
