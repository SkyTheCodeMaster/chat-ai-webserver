from __future__ import annotations

import asyncio
import tomllib
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import TYPE_CHECKING

import time
from .output import get_processor

if TYPE_CHECKING:
  pass

with open("config.toml") as f:
  config = tomllib.loads(f.read())


MODEL_ID = config["ai"]["model"]
DEVICE = config["ai"]["device"]
MODEL_LOCK = asyncio.Lock()

DEFAULT_TEMP = 0.4
DEFAULT_TOP_P = 0.92
DEFAULT_START_PROMPT = "You are a helpful assistant, with a focus on talking to people about relevant topics in the conversation."
OUTPUT_PROCESSOR = get_processor(MODEL_ID)


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
  temperature: float = DEFAULT_TEMP,
  top_p: float = DEFAULT_TOP_P,
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
        temperature=options.get("temperature", DEFAULT_TEMP),
        top_p=options.get("top_p", DEFAULT_TOP_P),
      ),
    )
    processed = OUTPUT_PROCESSOR(output_text)
  print(f"[{t()}] Finished generation.")
  return processed
