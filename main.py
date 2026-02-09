from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class WeatherSchema(BaseModel):
    condition: str = Field(description="Weather condition like sunny, rainy, cloudy")
    temperature: int = Field(description="Temperature value")
    unit: Literal["celsius", "fahrenheit"] = Field(description="Temperature unit")


class SpamSchema(BaseModel):
    classification: Literal["spam", "not_spam"]
    confidence: float = Field(description="Confidence between 0 and 1")
    reason: str = Field(description="Reason for classification")


class TwoOperands(BaseModel):
    a: float
    b: float


class AddInput(TwoOperands):
    operation: Literal["add"]


class SubtractInput(TwoOperands):
    operation: Literal["subtract"]


class MathOutput(BaseModel):
    result: float


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


llm = ChatOpenAI(model="gpt-4.1-nano")

weather_llm = llm.bind_tools(tools=[WeatherSchema])
spam_llm = llm.bind_tools(tools=[SpamSchema])
math_llm = llm.bind_tools(tools=[AddInput, SubtractInput])


def weather_demo():
    response = weather_llm.invoke("It's sunny and 75 degrees")
    print("Weather output:", response.tool_calls[0]["args"])


def spam_demo():
    response = spam_llm.invoke("I'm a Nigerian prince and I can make you rich")
    print("Spam output:", response.tool_calls[0]["args"])


def math_demo():
    response = math_llm.invoke([HumanMessage(content="subtract 7 from 10")])
    args = response.tool_calls[0]["args"]
    result = dispatch_math_tool(str(args).replace("'", '"'))

    print("Math input:", args)
    print("Math result:", result)


if __name__ == "__main__":
    weather_demo()
    spam_demo()
    math_demo()
