from langchain_core.messages import AIMessage
from template_tools import (
    continue_prompt_tool,
    display_tool,
    dummy_tool,
    model_selection_tool,
    user_input_tool,
)


class TemplateHandlers:
    """Handlers for TemplateApp workflow steps using tools.

    Manages input processing, tool operations, and result generation for
    the template app workflow including interactive capabilities.

    Attributes:
        llm: Initialized language model for processing.
        template_prompt: Prompt template for LLM processing.
        available_models: List of available LLM models.
        app: Reference to the TemplateApp instance.
    """

    def __init__(self, llm, template_prompt: str, available_models: list):
        """Initialize the handlers with LLM, prompt, and available models.

        Args:
            llm: Initialized language model (can be None initially).
            template_prompt: Prompt template string for LLM processing.
            available_models: List of available LLM model identifiers.
        """
        self.llm = llm
        self.template_prompt_template = template_prompt
        self.template_prompt = None
        self.available_models = available_models
        self.app = None

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
            if self.app:
                self.app.initialize_llm(selected_model)

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
            state["input_data"] = user_input

            # Display the user input
            print(f"📝 User Input: {user_input}")
            print("-" * 50)

            state["messages"].append(
                AIMessage(
                    content=(f"Received input: {len(user_input)} characters")
                )
            )

            return state

        except Exception as e:
            state["error"] = f"Error getting user input: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def process_input(self, state: dict) -> dict:
        """Process input data using the dummy tool and LLM.

        Args:
            state: Agent state with input data.

        Returns:
            dict: Updated state with processed data and result, or error.
        """
        try:
            print("🔧 Processing input data")
            print(f"📊 Input length: {len(state['input_data'])} characters")
            print("-" * 50)

            # First, use the dummy tool to "process" the input
            result = dummy_tool.invoke({"input_data": state["input_data"]})

            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Tool Error: {result['error']}")
                )
                return state

            # Store the tool result
            processed_data = result["processed_data"]
            state["processed_data"] = processed_data

            # Display the tool processing result
            print(f"🔧 Tool processed: {processed_data}")

            state["messages"].append(
                AIMessage(content="Tool processed input successfully")
            )

            # If LLM is available, use it for additional processing
            if self.llm and self.template_prompt:
                try:
                    prompt = self.template_prompt.invoke({
                        "input_data": state["processed_data"],
                    }).to_messages()

                    response = self.llm.invoke(prompt)
                    llm_result = response.content.strip()
                    state["result"] = llm_result

                    # Display the LLM processing result
                    print(f"🤖 LLM processed: {llm_result}")

                    state["messages"].append(
                        AIMessage(
                            content="LLM processed the input successfully"
                        )
                    )
                except Exception as e:
                    # If LLM fails, use the tool result as the final result
                    state["result"] = state["processed_data"]
                    print(f"⚠️ LLM failed, using tool result: {str(e)}")
                    state["messages"].append(
                        AIMessage(
                            content=(
                                "LLM processing failed, "
                                f"using tool result: {str(e)}"
                            )
                        )
                    )
            else:
                # No LLM available, use tool result as final result
                state["result"] = state["processed_data"]
                print("ℹ️ No LLM available, using tool result")
                state["messages"].append(
                    AIMessage(content="No LLM available, using tool result")
                )

            print("✅ Successfully processed input")
            return state

        except Exception as e:
            state["error"] = f"Error processing input: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            print(f"❌ Error: {str(e)}")
            return state

    def display_results(self, state: dict) -> dict:
        """Handle displaying results using the display tool.

        Args:
            state: Agent state with results to display.

        Returns:
            dict: Updated state after displaying results.
        """
        try:
            result = display_tool.invoke({
                "result": state.get("result", ""),
                "error": state.get("error", ""),
                "success": not bool(state.get("error")),
            })

            # Check if display was successful
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Display Error: {result['error']}")
                )
            else:
                display_success = result["displayed"]
                success_msg = (
                    "successfully " if display_success else "failed to be "
                )
                state["messages"].append(
                    AIMessage(
                        content=f"Results {success_msg}displayed to user"
                    )
                )

            return state

        except Exception as e:
            state["error"] = f"Error displaying results: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def ask_continue(self, state: dict) -> dict:
        """Handle asking user whether to continue using continue prompt tool.

        Args:
            state: Agent state.

        Returns:
            dict: Updated state with continue decision.
        """
        try:
            result = continue_prompt_tool.invoke({})

            if result["error"]:
                state["error"] = result["error"]
                state["continue_processing"] = False
                state["messages"].append(
                    AIMessage(
                        content=(f"Continue Prompt Error: {result['error']}")
                    )
                )
                return state

            should_continue = result["continue"]
            user_input = result["user_input"]

            state["continue_processing"] = should_continue
            state["user_choice"] = user_input
            state["messages"].append(
                AIMessage(
                    content=(
                        f"User chose to "
                        f"{'continue' if should_continue else 'exit'}"
                    )
                )
            )

            # Clear previous input and results for next iteration
            if should_continue:
                state["input_data"] = ""
                state["processed_data"] = ""
                state["result"] = ""
                state["error"] = None

            return state

        except Exception as e:
            state["error"] = f"Error asking to continue: {str(e)}"
            state["continue_processing"] = False
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state
