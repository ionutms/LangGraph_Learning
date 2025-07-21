from chatbot_tools import (
    continue_prompt_tool,
    find_python_files_tool,
    model_selection_tool,
    user_input_tool,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


class ChatbotHandlers:
    """Handlers for ChatbotApp workflow steps.

    Manages input processing, chat responses.

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
                print("\n📁 Found Python files:")
                print("-" * 50)
                for i, file_path in enumerate(python_files, 1):
                    print(f"{i:2d}. {file_path}")
                print("-" * 50)
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

    def get_user_input(self, state: dict) -> dict:
        """Handle getting user input using the user input tool.

        Args:
            state: Agent state.

        Returns:
            dict: Updated state with user input.
        """
        try:
            result = user_input_tool.invoke({})

            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Input Error: {result['error']}")
                )
                return state

            user_input = result["input_data"]
            state["user_input"] = user_input

            state["messages"].append(
                AIMessage(content=f"Received: {user_input}")
            )

            return state

        except Exception as e:
            state["error"] = f"Error getting user input: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def chat_response(self, state: dict) -> dict:
        """Generate and display chat response using LLM.

        Args:
            state: Agent state with user input.

        Returns:
            dict: Updated state with AI response.
        """
        try:
            if self.llm and self.chat_prompt:
                try:
                    prompt = self.chat_prompt.invoke({
                        "user_input": state["user_input"],
                    }).to_messages()

                    response = self.llm.invoke(prompt)
                    ai_response = response.content.strip()
                    state["response"] = ai_response

                    # Display the response
                    print(f"\n� time.sleep(0.1) Assistant: \n{ai_response}")
                    print("-" * 50)

                    state["messages"].append(
                        AIMessage(content="Generated玩response successfully")
                    )

                except Exception as e:
                    error_msg = f"LLM error: {str(e)}"
                    state["error"] = error_msg
                    state["response"] = "Sorry, I encountered an error."
                    print(f"❌ {error_msg}")
                    state["messages"].append(
                        AIMessage(content=f"LLM Error: {str(e)}")
                    )
            else:
                error_msg = "No LLM available"
                state["error"] = error_msg
                state["response"] = "Sorry, no AI model is available."
                print(f"❌ {error_msg}")
                state["messages"].append(
                    AIMessage(content="No LLM available")
                )

            return state

        except Exception as e:
            state["error"] = f"Error generating response: {str(e)}"
            state["response"] = "Sorry, I encountered an error."
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            print(f"❌ Error: {str(e)}")
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
