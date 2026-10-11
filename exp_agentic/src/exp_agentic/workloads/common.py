#! /usr/bin/env python


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build one Hermes-style tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


PATH = {"path": {"type": "string", "description": "Path relative to /work."}}

READ_FILE = tool("read_file", "Read a file from the working directory.", PATH, ["path"])

WRITE_FILE = tool(
    "write_file",
    "Write a file, replacing it if it exists.",
    {**PATH, "content": {"type": "string", "description": "The complete new contents."}},
    ["path", "content"],
)

LIST_FILES = tool("list_files", "List files under a directory.", PATH, ["path"])

DELETE_FILE = tool("delete_file", "Delete a file.", PATH, ["path"])

RUN_TESTS = tool("run_tests", "Run the test suite and return its output.", {}, [])

RUN_SCRIPT = tool(
    "run_script",
    "Run a Python script and return its output.",
    PATH,
    ["path"],
)

ASK_USER = tool(
    "ask_user",
    "Ask the user a question and wait for their answer.",
    {"question": {"type": "string", "description": "The question to ask."}},
    ["question"],
)

SUBMIT = tool(
    "submit",
    "Finish the task.",
    {"summary": {"type": "string", "description": "What you did."}},
    ["summary"],
)

GIVE_UP = tool(
    "give_up",
    "Give up on the task.",
    {"reason": {"type": "string", "description": "Why."}},
    ["reason"],
)

SYSTEM = """You are an agent working in a directory at /work.

You have tools to inspect and change files, and to run things. Work by making changes and checking \
them. When you are finished, call submit with a short summary of what you did.

Only call one tool at a time, and wait for its result before deciding what to do next."""
