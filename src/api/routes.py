from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.web import Response
from utils.authenticate import Approval, Key, authenticate, get_project_status
from utils.cors import add_cors_routes
from utils.limiter import Limiter
from utils.chat import generate_full_text, DEFAULT_TEMP, DEFAULT_TOP_P, DEFAULT_START_PROMPT

if TYPE_CHECKING:
  from utils.authenticate import User
  from utils.extra_request import Request
  from typing import Any

with open("config.toml") as f:
  config = tomllib.loads(f.read())
  frontend_version = config["pages"]["frontend_version"]
  exempt_ips = config["srv"]["ratelimit_exempt"]
  api_version = config["srv"]["api_version"]

limiter = Limiter(exempt_ips=exempt_ips)
routes = web.RouteTableDef()


@routes.get("/srv/get/")
@limiter.limit("60/m")
async def get_lp_get(request: Request) -> Response:
  packet = {
    "frontend_version": frontend_version,
    "api_version": api_version,
  }

  if request.app.POSTGRES_ENABLED:
    database_size_record = await request.conn.fetchrow(
      "SELECT pg_size_pretty ( pg_database_size ( current_database() ) );"
    )
    packet["db_size"] = database_size_record.get("pg_size_pretty", "-1 kB")

  return web.json_response(packet)


@routes.post("/chat/")
@limiter.limit("6/m")
async def post_chat(request: Request) -> Response:
  auth = await authenticate(request, cs=request.session)

  if isinstance(auth, Response):
    return auth
  else:
    if isinstance(auth, Key):
      # this means its a Key
      user: User = auth.user
    else:
      user: User = auth

    status = await get_project_status(user, "chat", cs=request.session)
    if status != Approval.APPROVED:
      return Response(
        status=401,
        text="please apply for project at https://auth.skystuff.cc/projects#chat",
      )

  body = await request.json()
  if "prompt" not in body:
    return Response(status=400, text="pass prompt in body!")
    
  prompt = body["prompt"]
  if "options" in body:
    body_options: dict[str,Any] = body["options"]
    options = {
      "max_tokens": body_options.get("max_tokens", 500),
      "temperature": body_options.get("temperature", DEFAULT_TEMP),
      "top_p": body_options.get("top_p", DEFAULT_TOP_P)
    }
  else:
    options = {
      "max_tokens": 500,
      "temperature": DEFAULT_TEMP,
      "top_p": DEFAULT_TOP_P
    }

  if "conversation" in body:
    # This is preceeding conversation, the user's new prompt
    # is added to this before feeding to the AI.
    conversation = body["conversation"]
  else:
    conversation = [
      {"role": "assistant", "content": DEFAULT_START_PROMPT}
    ]
  
  conversation.append({"role": "user", "content": prompt})

  try:
    response = await generate_full_text(conversation, options)
  except Exception as e:
    return Response(status=500, text=str(e))
  
  return web.json_response(response)

async def setup(app: web.Application) -> None:
  for route in routes:
    app.LOG.info(f"  ↳ {route}")
  app.add_routes(routes)
  add_cors_routes(routes, app)
