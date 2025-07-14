import os
import re
from typing import Annotated, Any, Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

LLM_MODEL = "groq:llama3-70b-8192"

LLM_INSTRUCTIONS = """
You are an expert SystemVerilog code generator.
Generate clean, well-structured SystemVerilog code based on user requirements.

**Coding Standards**:
  - Max 80 chars per line.
  - Use 2-space indent.
  - Include clear, relevant comments.
  - Use meaningful signal names.
  - Follow SystemVerilog best practices.

**Requirements**:
  1. **Design Module**:
     - Create a complete, compilable SystemVerilog module.
  2. **Testbench Module**:
     - Instantiate the design module.
     - Provide stimulus (e.g., clock, reset, inputs).
     - Include basic verification (e.g., output checks, result display).
     - Append '_tb' to module name for consistency.
     - Include VCD file generation with:
       - `$dumpfile("<module_name>_tb.vcd")` to save simulation data.
       - `$dumpvars(0, <module_name>_tb)` to dump all testbench variables.
    - Use a meaningful instance name for all instances.
    - Include a $finish statement to properly end the simulation.

**Output Format**:
  - Provide exactly two code blocks in order:
    - ```systemverilog design
      <Design module code here>
      ```
    - ```systemverilog testbench
      <Testbench module code here>
      ```
  - Ensure both blocks are present and correctly labeled.
  - Do not include other code blocks or markers.
"""


class AgentState(TypedDict):
    """
    State for SystemVerilog code generation agent.

    Attributes:
        user_request: User request for SystemVerilog code.
        generated_code: Generated SystemVerilog design code.
        testbench_code: Generated testbench code.
        env_content: Generated .env file content.
        messages: List of conversation messages.
        error: Any error messages during code generation.
        output_dir: Directory for saving files.
    """

    user_request: str
    generated_code: str
    testbench_code: str
    env_content: str
    messages: Annotated[list, add_messages]
    error: str
    output_dir: str


@tool
def generate_env_file_tool(generated_code: str) -> Dict[str, Any]:
    """
    Generate .env file content for the SystemVerilog project.

    Args:
        generated_code:
            Generated design code from which to extract module name.

    Returns:
        Dict: Contains env_content and any error message.
    """
    try:
        # Extract module name from design code
        module_match = re.search(r"module\s+(\w+)", generated_code)
        module_name = (
            module_match.group(1) if module_match else "generated_module"
        )
        design_filename = f"{module_name}.sv"
        testbench_filename = f"{module_name}_tb.sv"

        # Define .env key-value pairs
        env_lines = [
            f"PROJECT_DIR=./{module_name}",
            f"DESIGN_FILE={design_filename}",
            f"TESTBENCH_FILE={testbench_filename}",
            f"TOP_MODULE={module_name}_tb",
            f"VCD_FILE={module_name}_tb.vcd",
        ]

        # Join lines with newlines for .env content
        env_content = "\n".join(env_lines)

        return {"env_content": env_content, "error": ""}
    except Exception as error:
        return {
            "env_content": "",
            "error": f"Environment file generation error: {str(error)}",
        }


@tool
def save_code_tool(
    design_code: str,
    testbench_code: str,
    env_content: str,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """
    Save SystemVerilog code and .env file to a module-specific folder.

    Args:
        design_code: Generated design code.
        testbench_code: Generated testbench code.
        env_content: Generated .env file content.
        output_dir: Base directory for saving files (default: 'output').

    Returns:
        Dict: Contains list of status messages and any error message.
    """
    try:
        messages = []
        module_match = re.search(r"module\s+(\w+)", design_code)
        module_name = (
            module_match.group(1) if module_match else "generated_module"
        )

        # Create module-specific directory
        module_dir = os.path.join(output_dir, module_name)
        os.makedirs(module_dir, exist_ok=True)

        # Save design code
        design_filename = f"{module_name}.sv"
        design_filepath = os.path.join(module_dir, design_filename)
        with open(design_filepath, "w") as design_file:
            design_file.write(design_code)
        messages.append(f"Design saved to {design_filepath}")

        # Save testbench code if available
        testbench_filename = f"{module_name}_tb.sv"
        testbench_filepath = os.path.join(module_dir, testbench_filename)
        if testbench_code:
            with open(testbench_filepath, "w") as testbench_file:
                testbench_file.write(testbench_code)
            messages.append(f"Testbench saved to {testbench_filepath}")
        else:
            messages.append("No testbench code to save")

        # Save .env file
        env_filepath = os.path.join(module_dir, ".env")
        if env_content:
            with open(env_filepath, "w") as env_file:
                env_file.write(env_content)
            messages.append(f"Environment file saved to {env_filepath}")
        else:
            messages.append("No .env file content to save")

        return {"messages": messages, "error": ""}
    except Exception as error:
        return {"messages": [], "error": f"Error saving code: {str(error)}"}


class SystemVerilogCodeGenerator:
    def __init__(self):
        """
        Initialize SystemVerilog code generator with LLM.

        Generates SystemVerilog code and testbench per standards.
        """
        self.tools = [generate_env_file_tool, save_code_tool]
        self.llm = init_chat_model(LLM_MODEL).bind_tools(self.tools)
        self.sv_prompt = ChatPromptTemplate.from_messages([
            ("system", LLM_INSTRUCTIONS),
            ("human", "{user_request}"),
        ])
        self.graph = self._create_graph()

    def _create_graph(self):
        """
        Create LangGraph workflow for code generation.

        Returns:
            StateGraph: Compiled workflow with nodes and edges.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("generate_env", self._call_generate_env_tool)
        workflow.add_node("save_code", self._call_save_code_tool)
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "generate_env")
        workflow.add_edge("generate_env", "save_code")
        workflow.add_edge("save_code", END)
        return workflow.compile()

    def _generate_code(self, state: AgentState) -> AgentState:
        """
        Generate SystemVerilog code and testbench from user request.

        Args:
            state: Current state with user request and data.

        Returns:
            AgentState: Updated with generated code, testbench, messages.
        """
        try:
            messages = [
                {"role": "system", "content": LLM_INSTRUCTIONS},
                {"role": "user", "content": state["user_request"]},
            ]
            response = self.llm.invoke(messages)
            content = response.content

            # Extract code blocks
            design_code = ""
            testbench_code = ""
            code_blocks = re.findall(
                r"```systemverilog\s*(.*?)\s*```", content, re.DOTALL
            )
            if len(code_blocks) >= 2:
                design_code = code_blocks[0].strip()
                testbench_code = code_blocks[1].strip()
            elif len(code_blocks) == 1:
                design_code = code_blocks[0].strip()
                state["error"] = "Testbench not provided by LLM"
            else:
                state["error"] = "No valid SystemVerilog code found"
                return state

            state["generated_code"] = design_code
            state["testbench_code"] = testbench_code
            state["messages"].append(
                AIMessage(
                    content=f"Generated SystemVerilog design and testbench "
                    f"for: {state['user_request']}"
                )
            )
        except Exception as error:
            state["error"] = f"Code generation error: {str(error)}"
        return state

    def _call_generate_env_tool(self, state: AgentState) -> AgentState:
        """
        Call the generate_env_file_tool with the generated code.

        Args:
            state: Current state with generated code and data.

        Returns:
            AgentState: Updated with .env file content from tool.
        """
        try:
            result = generate_env_file_tool.invoke({
                "generated_code": state["generated_code"]
            })
            state["env_content"] = result["env_content"]
            if result["error"]:
                state["error"] = result["error"]
            else:
                module_name = (
                    re.search(
                        r"module\s+(\w+)", state["generated_code"]
                    ).group(1)
                    if re.search(r"module\s+(\w+)", state["generated_code"])
                    else "generated_module"
                )
                state["messages"].append(
                    AIMessage(
                        content=f"Generated .env file for {module_name}"
                    )
                )
        except Exception as error:
            state["error"] = f"Error calling env tool: {str(error)}"
        return state

    def _call_save_code_tool(self, state: AgentState) -> AgentState:
        """
        Call the save_code_tool with generated code and env content.

        Args:
            state:
                Current state with generated code, testbench, and env content.

        Returns:
            AgentState: Updated with save status messages from tool.
        """
        try:
            result = save_code_tool.invoke({
                "design_code": state["generated_code"],
                "testbench_code": state["testbench_code"],
                "env_content": state["env_content"],
                "output_dir": state["output_dir"],
            })
            for message in result["messages"]:
                state["messages"].append(AIMessage(content=message))
            if result["error"]:
                state["error"] = result["error"]
        except Exception as error:
            state["error"] = f"Error calling save code tool: {str(error)}"
        return state

    def generate(
        self,
        user_request: str,
        save_file: bool = False,
        output_dir: str = "output",
    ) -> Dict[str, Any]:
        """
        Generate SystemVerilog design, testbench, and .env file.

        Args:
            user_request: Desired SystemVerilog module description.
            save_file: Save generated code to files (default: False).
            output_dir: Directory for saving files (default: 'output').

        Returns:
            Dict:
                Contains success, design_code, testbench_code, env_content,
                error, messages.
        """
        initial_state = {
            "user_request": user_request,
            "generated_code": "",
            "testbench_code": "",
            "env_content": "",
            "messages": [HumanMessage(content=user_request)],
            "error": "",
            "output_dir": output_dir,
        }
        if save_file:
            # Ensure output directory exists
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as error:
                initial_state["error"] = (
                    f"Failed to create output directory: {str(error)}"
                )
                return {
                    "success": False,
                    "design_code": "",
                    "testbench_code": "",
                    "env_content": "",
                    "error": initial_state["error"],
                    "messages": initial_state["messages"],
                }
            result = self.graph.invoke(initial_state)
        else:
            # Skip save_code node if not saving files
            workflow = StateGraph(AgentState)
            workflow.add_node("generate_code", self._generate_code)
            workflow.add_node("generate_env", self._call_generate_env_tool)
            workflow.add_edge(START, "generate_code")
            workflow.add_edge("generate_code", "generate_env")
            workflow.add_edge("generate_env", END)
            result = workflow.compile().invoke(initial_state)

        return {
            "success": not bool(result["error"]),
            "design_code": result["generated_code"],
            "testbench_code": result["testbench_code"],
            "env_content": result["env_content"],
            "error": result["error"],
            "messages": result["messages"],
        }


if __name__ == "__main__":
    generator = SystemVerilogCodeGenerator()

    print("SystemVerilog Code Generator Graph:")
    print(generator.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60)

    # Example requests
    test_requests = [
        "Create a simple 4-bit counter module",
        "Generate a 2-to-1 multiplexer with enable signal",
        "Create a D flip-flop with asynchronous reset",
    ]

    for request in test_requests:
        print(f"\nRequest: {request}")
        print("=" * 60)

        result = generator.generate(
            request, save_file=True, output_dir="sv_output"
        )

        if result["success"]:
            print("✅ Design and testbench generation successful")
        else:
            print(f"❌ Error: {result['error']}")

        print("\n" + "=" * 60)
