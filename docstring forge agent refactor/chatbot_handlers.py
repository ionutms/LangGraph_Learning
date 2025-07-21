from chatbot_tools import (
    continue_prompt_tool,
    find_python_files_tool,
    model_selection_tool,
    select_file_tool,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


class ChatbotHandlers:
    """Handlers for ChatbotApp workflow steps.

    Manages file selection, input processing, and chat responses.

    Attributes:
        llm: Initialized language model.
        chat_prompt: Prompt template for LLM.
        available_models: List of available LLM models.
    """

    def __init__(self, chat_prompt_template: str, available_models: list):
        """Initialize the handlers.

        Args:
            chat_prompt_template: Prompt template string for LLM.
            available_models: List of available LLM model identifiers.
        """
        self.llm = None
        self.chat_prompt_template = chat_prompt_template
        self.chat_prompt = None
        self.available_models = available_models
        self.selected_model = ""

    def find_files(self, state: dict) -> dict:
        """Find Python files in the current directory.

        Args:
            state: Agent state.

        Returns:
            dict: Updated state with list of Python files.
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

            python_files = result["python_files"]
            state["python_files"] = python_files

            if python_files:
                state["messages"].append(
                    AIMessage(
                        content=f"Found {len(python_files)} Python files"
                    )
                )
            else:
                print("\n📁 No Python files found.")
                state["messages"].append(
                    AIMessage(content="No Python files found")
                )

            return state

        except Exception as e:
            state["error"] = f"Error finding files: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def select_file(self, state: dict) -> dict:
        """Select a Python file from the list of found files.

        Args:
            state: Agent state with list of Python files.

        Returns:
            dict: Updated state with selected file or error.
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
                        content=f"File Selection Error: {result['error']}"
                    )
                )
                return state

            selected_file = result["selected_file"]
            state["selected_file"] = selected_file
            state["messages"].append(
                AIMessage(content=f"Selected file: {selected_file}")
            )

            return state

        except Exception as e:
            state["error"] = f"Error selecting file: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def initialize_llm(self, model: str):
        """Initialize LLM with selected model.

        Args:
            model: LLM model identifier to initialize.
        """
        self.llm = init_chat_model(model, temperature=0.0, max_tokens=4000)
        self.selected_model = model
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.chat_prompt_template),
            ("user", "User message: {user_input}"),
        ])

    def select_model(self, state: dict) -> dict:
        """Handle model selection using the model selection tool.

        Args:
            state: Agent state.

        Returns:
            dict: Updated state with selected model.
        """
        try:
            result = model_selection_tool.invoke({
                "available_models": self.available_models,
                "current_model": state.get("selected_model", ""),
            })

            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(
                        content=f"Model Selection Error: {result['error']}"
                    )
                )
                return state

            selected_model = result["selected_model"]
            state["selected_model"] = selected_model
            state["messages"].append(
                AIMessage(content=f"Selected model: {selected_model}")
            )

            # Initialize LLM with selected model
            self.initialize_llm(selected_model)

            return state

        except Exception as e:
            state["error"] = f"Error selecting model: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def ask_continue(self, state: dict) -> dict:
        """Handle asking user whether to continue chatting.

        Args:
            state: Agent state.

        Returns:
            dict: Updated state with continue decision.
        """
        try:
            result = continue_prompt_tool.invoke({})

            if result["error"]:
                state["error"] = result["error"]
                state["continue_chatting"] = False
                state["messages"].append(
                    AIMessage(
                        content=f"Continue Prompt Error: {result['error']}"
                    )
                )
                return state

            should_continue = result["continue"]
            user_input = result["user_input"]

            state["continue_chatting"] = should_continue
            state["user_choice"] = user_input
            state["messages"].append(
                AIMessage(
                    content=(
                        f"User chose to "
                        f"{'continue' if should_continue else 'exit'}"
                    )
                )
            )

            # Clear previous input and response for next iteration
            if should_continue:
                state["user_input"] = ""
                state["response"] = ""
                state["error"] = None

            return state

        except Exception as e:
            state["error"] = f"Error asking to continue: {str(e)}"
            state["continue_chatting"] = False
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state
