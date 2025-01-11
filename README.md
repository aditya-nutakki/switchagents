# Switchagents
Develop ANY agent. Written in pure Python (and its native libraries), no langchain, no llamaindex !

## Usage:
- Define tools, system prompts etc in your file
- Switchagents fundamentally works on "bundles" - which is essentially a way to package tools, system prompts, temperature and other parameters into a single place.
- Create your bundle. Each bundle must contain the model (str), available models (list), temperature (int), id (str), system_prompt (str), base_url (str), API_KEY (str), tools (dependant on the schema being used, refer to [openai docs]([url](https://platform.openai.com/docs/guides/function-calling)) and [anthropic docs]([url](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)) documentation for more information
- Each bundle is of the following type:
  ```
    bundle = {
      "id": "my-unique-bundle",
      "model" : "claude-3-haiku-20240307",
      "system_prompt": my_system_prompt,
      "tools" : [],
      "temperature": 0.2,
      "stream_def": None,
      "available_models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
      "base_url": "", # defaults to whatever the client/sdk has
      "key_env_variable": "ANTHROPIC_API_KEY"
  }

- Pass the bundle into the agent and now you have a module which can recursively be called until the job at hand is finished (especially works well for multi-hop queries)

## Features:
- Supports streaming with function calling
- Modular, PyTorch-ish usage
- Supports Local-LLM's (via vLLM)
- Easily build multi-agent systems

## Upcoming Features:
- Support for Vector stores
- Support for embedding models
