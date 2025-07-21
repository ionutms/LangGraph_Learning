from chatbot_tools import (
    extract_docstrings_tool,
    find_python_files_tool,
    load_file_tool,
    model_selection_tool,
    remove_docstrings_and_comments_tool,
    save_file_tool,
    select_file_tool,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


class ChatbotHandlers:
    """Handle ChatbotApp workflow steps for file and docstring processing.

    Manages file selection, action selection, docstring processing,
    and LLM updates.

    Attributes:
        llm: Language model for chat and docstring updates.
        chat_prompt: Prompt template for chat responses.
        doc_prompt: Prompt template for docstring updates.
        available_models: List of LLM model identifiers.
    """

    def __init__(self, chat_prompt_template: str, available_models: list):
        """Initialize handlers with prompt and models.

        Args:
            chat_prompt_template: Template for LLM chat responses.
            available_models: List of LLM model identifiers.
        """
        self.llm = None
        self.chat_prompt_template = chat_prompt_template
        self.chat_prompt = None
        self.doc_prompt_template = """
You are an expert Python docstring generator.
- Improve/add docstrings in code, using Google style, no examples.
- Max 79 chars per line.
- Remove trailing whitespace from docstrings.
- Add docstrings to undocumented functions/classes.
- Do not modify non-docstring code.

**Input**:
- Docstrings: {docstrings_info}
- Code:
```python
{original_code}
```

**Output**:
- Updated Python code with new/improved docstrings.
- Use single ```python block.
"""
        self.doc_prompt = None
        self.available_models = available_models
        self.selected_model = ""

    def find_files(self, state: dict) -> dict:
        """Find Python files in the current directory.

        Args:
            state: Agent state.

        Returns:
            dict: State with list of Python files.
        """
        try:
            result = find_python_files_tool.invoke({"directory": "."})
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(
                        content=f"Error finding files: {result['error']}"
                    )
                )
                return state
            state["python_files"] = result["python_files"]
            if state["python_files"]:
                state["messages"].append(
                    AIMessage(
                        content=f"Found {len(state['python_files'])} files"
                    )
                )
            else:
                print("\n📁 No Python files found.")
                state["messages"].append(
                    AIMessage(content="No Python files found")
                )
            return state
        except Exception as e:
            state["error"] = f"Error finding files: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def select_file(self, state: dict) -> dict:
        """Select a Python file from found files.

        Args:
            state: Agent state with Python files.

        Returns:
            dict: State with selected file or error.
        """
        try:
            if state.get("error"):
                return state
            result = select_file_tool.invoke({
                "python_files": state["python_files"]
            })
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(
                        content=f"File selection error: {result['error']}"
                    )
                )
                return state
            state["selected_file"] = result["selected_file"]
            state["messages"].append(
                AIMessage(content=f"Selected file: {result['selected_file']}")
            )
            return state
        except Exception as e:
            state["error"] = f"Error selecting file: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def select_action(self, state: dict) -> dict:
        """Prompt user to select an action for the file.

        Args:
            state: Agent state with selected file.

        Returns:
            dict: State with selected action or error.
        """
        try:
            if state.get("error") or not state.get("selected_file"):
                return state
            actions = {"r": "remove", "u": "update", "q": "quit"}
            print("\n🔧 Actions:")
            print("  r - Remove docstrings/comments")
            print("  u - Update docstrings with LLM")
            print("  q - Quit")
            while True:
                act = input("\nSelect action: ").lower().strip()
                if act not in actions:
                    print("❌ Invalid action. Use r, u, or q.")
                    continue
                sel = actions[act]
                if sel == "quit":
                    state["continue_chatting"] = False
                    state["messages"].append(
                        AIMessage(content="User chose to quit")
                    )
                    return state
                state["action"] = sel
                state["output_dir"] = "processed_files"
                state["messages"].append(
                    AIMessage(content=f"Selected action: {sel}")
                )
                return state
        except KeyboardInterrupt:
            state["error"] = "Action selection cancelled"
            state["continue_chatting"] = False
            state["messages"].append(
                AIMessage(content="Action selection cancelled")
            )
            return state
        except Exception as e:
            state["error"] = f"Error selecting action: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def initialize_llm(self, model: str):
        """Initialize LLM with selected model.

        Args:
            model: LLM model identifier.
        """
        self.llm = init_chat_model(model, temperature=0.0, max_tokens=4000)
        self.selected_model = model
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.chat_prompt_template),
            ("user", "User message: {user_input}"),
        ])
        self.doc_prompt = ChatPromptTemplate.from_messages([
            ("system", self.doc_prompt_template),
            (
                "user",
                "{docstrings_info}\n\nCode:\n```python\n{original_code}\n```",
            ),
        ])

    def select_model(self, state: dict) -> dict:
        """Handle model selection for LLM updates.

        Args:
            state: Agent state.

        Returns:
            dict: State with selected model.
        """
        try:
            if state.get("action") != "update":
                return state
            result = model_selection_tool.invoke({
                "models": self.available_models,
                "current": state.get("selected_model", ""),
            })
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(
                        content=f"Model selection error: {result['error']}"
                    )
                )
                return state
            state["selected_model"] = result["selected_model"]
            state["messages"].append(
                AIMessage(
                    content=f"Selected model: {result['selected_model']}"
                )
            )
            self.initialize_llm(result["selected_model"])
            return state
        except Exception as e:
            state["error"] = f"Error selecting model: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def load_file(self, state: dict) -> dict:
        """Load the selected Python file.

        Args:
            state: Agent state with selected file.

        Returns:
            dict: State with loaded code or error.
        """
        try:
            if state.get("error"):
                return state
            result = load_file_tool.invoke({
                "file_path": state["selected_file"]
            })
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Error: {result['error']}")
                )
            else:
                state["original_code"] = result["file_content"]
                state["processed_code"] = result["file_content"]
                state["messages"].append(
                    AIMessage(
                        content=f"Loaded file: {state['selected_file']}"
                    )
                )
            return state
        except Exception as e:
            state["error"] = f"Error loading file: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def analyze_docstrings(self, state: dict) -> dict:
        """Extract docstring info from code.

        Args:
            state: Agent state with original code.

        Returns:
            dict: State with docstring info or error.
        """
        try:
            if state.get("error"):
                return state
            result = extract_docstrings_tool.invoke({
                "code": state["original_code"]
            })
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Error: {result['error']}")
                )
            else:
                state["docstring_info"] = result["docstring_info"]
                state["messages"].append(
                    AIMessage(
                        content=f"Found {len(result['docstring_info'])} "
                        "docstrings"
                    )
                )
            return state
        except Exception as e:
            state["error"] = f"Error analyzing docstrings: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def process_docstrings(self, state: dict) -> dict:
        """Process docstrings based on action.

        Args:
            state: Agent state with action and code.

        Returns:
            dict: State with processed code or LLM prompt.
        """
        try:
            if state.get("error"):
                return state
            action = state["action"]
            if action == "remove":
                result = remove_docstrings_and_comments_tool.invoke({
                    "code": state["original_code"]
                })
                if result["error"]:
                    state["error"] = result["error"]
                    state["messages"].append(
                        AIMessage(content=f"Error: {result['error']}")
                    )
                else:
                    state["processed_code"] = result["processed_code"]
                    state["messages"].append(
                        AIMessage(content="Removed docstrings and comments")
                    )
                return state
            elif action == "update":
                if not self.llm or not self.doc_prompt:
                    state["error"] = "LLM not initialized"
                    state["messages"].append(
                        AIMessage(content="Error: LLM not initialized")
                    )
                    return state
                info = "\n".join([
                    f"- {d['type']} '{d['name']}' (line {d['lineno']}): "
                    f"{d['docstring'][:50]}..."
                    if d["docstring"]
                    else "None"
                    for d in state["docstring_info"]
                ])
                prompt = self.doc_prompt.invoke({
                    "docstrings_info": info,
                    "original_code": state["original_code"],
                }).to_messages()
                state["messages"] = prompt
                state["messages"].append(
                    AIMessage(
                        content="Prepared LLM prompt for docstring update"
                    )
                )
                return state
            state["error"] = f"Unknown action: {action}"
            state["messages"].append(
                AIMessage(content=f"Error: Unknown action {action}")
            )
            return state
        except Exception as e:
            state["error"] = f"Error processing docstrings: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def llm_process(self, state: dict) -> dict:
        """Update docstrings using LLM.

        Args:
            state: Agent state with LLM prompt.

        Returns:
            dict: State with updated code or error.
        """
        try:
            if state.get("error") or not state.get("messages"):
                return state
            if not self.llm:
                state["error"] = "LLM not initialized"
                state["messages"].append(
                    AIMessage(content="Error: LLM not initialized")
                )
                return state
            resp = self.llm.invoke(state["messages"])
            content = resp.content
            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                code = (
                    content[start:end].strip()
                    if end != -1
                    else content[start:].strip()
                )
            else:
                code = content.strip()
            state["processed_code"] = code
            state["messages"].append(
                AIMessage(content="Updated docstrings with LLM")
            )
            return state
        except Exception as e:
            state["error"] = f"Error in LLM processing: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def save_result(self, state: dict) -> dict:
        """Save processed code to output directory.

        Args:
            state: Agent state with processed code.

        Returns:
            dict: State with saved file path or error.
        """
        try:
            if state.get("error"):
                return state
            result = save_file_tool.invoke({
                "content": state["processed_code"],
                "output_dir": state["output_dir"],
                "orig_path": state["selected_file"],
            })
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Error: {result['error']}")
                )
            else:
                state["saved_file"] = result["saved_file"]
                state["messages"].append(
                    AIMessage(content=f"Saved file to {result['saved_file']}")
                )
            return state
        except Exception as e:
            state["error"] = f"Error saving file: {e}"
            state["messages"].append(AIMessage(content=f"Error: {e}"))
            return state

    def should_use_llm(self, state: dict) -> str:
        """Check if LLM processing is needed.

        Args:
            state: Agent state with action.

        Returns:
            str: 'use_llm' or 'skip_llm'.
        """
        if state.get("error"):
            return "skip_llm"
        return "use_llm" if state["action"] == "update" else "skip_llm"
