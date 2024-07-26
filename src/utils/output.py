from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from typing import Callable

SMOLLM_PATTERN = re.compile(
  r"<\|im_start\|>(.*?)<\|im_end\|>", re.MULTILINE | re.DOTALL
)
TINYLLAMA_PATTERN = re.compile(
  r"<s>(.*?)</s>", re.MULTILINE | re.DOTALL
)

def process_smollm(output_text: str) -> dict:
  """
  Input:
  <|im_start|>system\nYou are a friendly chatbot.<|im_end|>\n<|im_start|>user\nSkyCrafter0: Hello there! What is my name?<|im_end|>\n<|im_start|>assistant\nYour name is SkyCrafter0.<|im_end|>

  Steps:
  extract data between <|im_start|> and <|im_end|>
  first line is the "role", everything else is content
  take last content as "response"
  Outputs something like:
  {
    "response": "latest response here",
    "conversation": [
      {"role":"assistant","content":"message"}
    ]
  }"""
  matches: list[str] = SMOLLM_PATTERN.findall(output_text)
  if not matches:
    raise Exception("No output data found!")

  conversation = []
  for match in matches:
    split = match.split("\n")
    role = split.pop(0)
    content = "\n".join(split)
    conversation.append({"role": role, "content": content})

  response = conversation[-1]["content"]
  output_response = {"response": response, "conversation": conversation}
  return output_response

def process_tinyllama(output_text: str) -> dict:
  """
  Input:
  <s> <|assistant|>\nYou are a helpful assistant, with a focus on talking to people about relevant topics in the conversation.</s> \n<|user|>\nSkyCrafter0: What is my name right now?</s> \n<|assistant|>\nSkyCrafter0: Your name is SkyCrafter0.</s>

  Steps:
  extract data between <s> and </s>
  first line is the "role", strip <|>, everything else is content
  take last content as "response"
  Outputs something like:
  {
    "response": "latest response here",
    "conversation": [
      {"role":"assistant","content":"message"}
    ]
  }"""
  matches: list[str] = TINYLLAMA_PATTERN.findall(output_text)
  if not matches:
    raise Exception("No output data found!")

  conversation = []
  for match in matches:
    split = match.split("\n")
    role = split.pop(0)
    role = role.strip(" <|>")
    content = ("\n".join(split)).strip()
    conversation.append({"role": role, "content": content})

  response = conversation[-1]["content"]
  output_response = {"response": response, "conversation": conversation}
  return output_response

def process_default(output_text: str) -> dict:
  "Return basic output"
  return {"response": output_text, "conversation": []}

OUTPUT_PROCESSORS: dict[str, Callable[[str], dict]] = {
  "HuggingFaceTB/SmolLM-1.7B-Instruct": process_smollm,
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0": process_tinyllama
}

def get_processor(model_name: str) -> Callable[[str], dict]:
  return OUTPUT_PROCESSORS.get(model_name, process_default)