from __future__ import annotations

import asyncio
import tomllib
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import TYPE_CHECKING

import time

if TYPE_CHECKING:
  pass

with open("config.toml") as f:
  config = tomllib.loads(f.read())

OUTPUT_PATTERN = re.compile(
  r"<\|im_start\|>(.*?)<\|im_end\|>", re.MULTILINE | re.DOTALL
)
MODEL_ID = config["ai"]["model"]
DEVICE = config["ai"]["device"]
MODEL_LOCK = asyncio.Lock()


if not torch.cuda.is_available() and "cuda" in DEVICE:
  raise Exception("Cuda is not available for model inference!")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model = model.to(DEVICE)


def setup_chat(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
  out = []
  for message in conversation:
    # Check for missing kwargs...
    if not message.get("role", None):
      raise Exception("Missing role in conversation!")
    if not message.get("content", None):
      raise Exception("Missing content in conversation!")

    out.append({"role": message["role"], "content": message.get("content")})

  return out


def generate_text(
  _conversation: list[dict[str, str]],
  *,
  max_new_tokens: int = 500,
  temperature: float = 0.6,
  top_p: float = 0.92,
  do_sample: bool = True,
) -> dict[str, str | list[dict[str, str]]]:
  conversation = setup_chat(_conversation)

  max_new_tokens = min(max_new_tokens, 500)

  input_text = tokenizer.apply_chat_template(conversation, tokenize=False)
  inputs = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
  outputs = model.generate(
    inputs,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=do_sample,
  )
  output_text = tokenizer.decode(outputs[0])
  return output_text


def process_output(output_text: str) -> dict:
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
  matches: list[str] = OUTPUT_PATTERN.findall(output_text)
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


async def generate_full_text(
  conversation: list[dict[str, str]], options: dict
) -> dict:
  loop = asyncio.get_running_loop()
  start = time.time()

  def t():
    return round(time.time() - start, 4)

  print(f"[{t()}] Waiting for lock...")
  async with MODEL_LOCK:
    print(f"[{t()}] Lock acquired, generating text...")
    output_text = await loop.run_in_executor(
      None,
      lambda: generate_text(
        conversation,
        max_new_tokens=options.get("max_new_tokens", 500),
        temperature=options.get("temperature", 0.6),
        top_p=options.get("top_p", 0.92),
      ),
    )
    processed = process_output(output_text)
  print(f"[{t()}] Finished generation.")
  return processed
