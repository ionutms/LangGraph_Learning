from langchain_core.messages import AIMessage
from template_tools import dummy_tool


class TemplateHandlers:
    """Handlers for TemplateApp workflow steps using tools.

    Manages input processing, tool operations, and result generation for
    the template app workflow.

    Attributes:
        llm: Initialized language model for processing.
        template_prompt: Prompt template for LLM processing.
    """

    def __init__(self, llm, template_prompt: str):
        """Initialize the handlers with LLM and prompt.

        Args:
            llm: Initialized language model (can be None initially).
            template_prompt: Prompt template string for LLM processing.
        """
        self.llm = llm
        self.template_prompt_template = template_prompt
        self.template_prompt = None

    def process_input(self, state: dict) -> dict:
        """Process input data using the dummy tool and LLM.

        Args:
            state: Agent state with input data.

        Returns:
            dict: Updated state with processed data and result, or error.
        """
        try:
            # First, use the dummy tool to "process" the input
            tool_result = dummy_tool.invoke({
                "input_data": state["input_data"]
            })

            if tool_result["error"]:
                state["error"] = tool_result["error"]
                state["messages"].append(
                    AIMessage(content=f"Tool Error: {tool_result['error']}")
                )
                return state

            # Store the tool result
            state["processed_data"] = tool_result["processed_data"]
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
                    state["result"] = response.content.strip()
                    state["messages"].append(
                        AIMessage(
                            content="LLM processed the input successfully"
                        )
                    )
                except Exception as e:
                    # If LLM fails, use the tool result as the final result
                    state["result"] = state["processed_data"]
                    state["messages"].append(
                        AIMessage(
                            content="LLM processing failed, "
                            f"using tool result: {str(e)}"
                        )
                    )
            else:
                # No LLM available, use tool result as final result
                state["result"] = state["processed_data"]
                state["messages"].append(
                    AIMessage(content="No LLM available, using tool result")
                )

            return state

        except Exception as e:
            state["error"] = f"Error processing input: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state
