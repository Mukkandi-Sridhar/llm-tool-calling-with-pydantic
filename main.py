from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# -------------------------------
# 1. WEATHER TOOL SCHEMA
# -------------------------------

class WeatherSchema(BaseModel):
    condition: str = Field(description="Weather condition like sunny, rainy, cloudy")
    temperature: int = Field(description="Temperature value")
    unit: Literal["celsius", "fahrenheit"] = Field(description="Temperature unit")


# -------------------------------
# 2. SPAM CLASSIFIER SCHEMA
# -------------------------------

class SpamSchema(BaseModel):
    classification: Literal["spam", "not_spam"]
    confidence: float = Field(description="Confidence between 0 and 1")
    reason: str = Field(description="Reason for classification")


# -------------------------------
# 3. MATH TOOL SCHEMAS
# -------------------------------

class TwoOperands(BaseModel):
    a: float
    b: float


class AddInput(TwoOperands):
    operation: Literal["add"]


class SubtractInput(TwoOperands):
    operation: Literal["subtract"]


class MathOutput(BaseModel):
    result: float


# -------------------------------
# 4. TOOL FUNCTIONS
# -------------------------------

def add_tool(data: AddInput) -> MathOutput:
    return MathOutput(result=data.a + data.b)


def subtract_tool(data: SubtractInput) -> MathOutput:
    return MathOutput(result=data.a - data.b)


def dispatch_math_tool(json_payload: str) -> str:
    base = SubtractInput.parse_raw(json_payload)

    if base.operation == "add":
        output = add_tool(AddInput.parse_raw(json_payload))
    elif base.operation == "subtract":
        output = subtract_tool(SubtractInput.parse_raw(json_payload))
    else:
        raise ValueError("Unsupported operation")

    return output.json()


# -------------------------------
# 5. LLM SETUP
# -------------------------------

llm = ChatOpenAI(model="gpt-4.1-nano")

weather_llm = llm.bind_tools(tools=[WeatherSchema])
spam_llm = llm.bind_tools(tools=[SpamSchema])
math_llm = llm.bind_tools(tools=[AddInput, SubtractInput])


# -------------------------------
# 6. DEMOS
# -------------------------------

def weather_demo():
    response = weather_llm.invoke("It's sunny and 75 degrees")
    print("Weather Tool Output:", response.tool_calls[0]["args"])


def spam_demo():
    response = spam_llm.invoke("I'm a Nigerian prince and I can make you rich")
    print("Spam Tool Output:", response.tool_calls[0]["args"])


def math_demo():
    response = math_llm.invoke([HumanMessage(content="subtract 7 from 10")])
    args = response.tool_calls[0]["args"]
    result_json = dispatch_math_tool(json_payload=str(args).replace("'", '"'))
    print("Math Tool Input:", args)
    print("Math Tool Output:", result_json)


# -------------------------------
# 7. ENTRY POINT
# -------------------------------

if __name__ == "__main__":
    print("\n--- WEATHER DEMO ---")
    weather_demo()

    print("\n--- SPAM DEMO ---")
    spam_demo()

    print("\n--- MATH DEMO ---")
    math_demo()
