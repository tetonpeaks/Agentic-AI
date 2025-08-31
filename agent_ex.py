from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
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

# Define schema for structured output
class MaterialAnalysis(BaseModel):
    material: str = Field(description="Name of the material")
    max_temperature: int = Field(description="Maximum operating temperature in Celsius")
    notes: str = Field(description="Additional considerations")

# Create parser
parser = PydanticOutputParser(pydantic_object=MaterialAnalysis)

# Get the format instructions from the parser
format_instructions = parser.get_format_instructions()

# Define LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a materials analysis assistant.
    Provide structured output about the given material.

    Material: {material}
    Required temperature: {temp_req}C

    {format_instructions}
    """
)

# Chain (new style using | instead of deprecated LLMChain)
chain = prompt | llm | parser

# Run it
result = chain.invoke({
    "material": "Aluminum 7075",
    "temp_req": 250,
    "format_instructions": format_instructions
})

print(f"{bcolors.OKGREEN}response: {bcolors.ENDC}{result}")
