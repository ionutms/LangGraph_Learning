from dotenv import load_dotenv
from typing import Dict, Any, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
import re
import os

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
    - First: ```systemverilog design
      <Design module code here>
      ```
    - Second: ```systemverilog testbench
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
      messages: List of conversation messages.
      error: Any error messages during code generation.
    """

    user_request: str
    generated_code: str
    testbench_code: str
    messages: Annotated[list, add_messages]
    error: str


class SystemVerilogCodeGenerator:
    def __init__(self):
        """
        Initialize SystemVerilog code generator with LLM.

        Generates SystemVerilog code and testbench per standards.
        """
        self.llm = init_chat_model(LLM_MODEL)
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
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", END)
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
            print(f"Raw LLM response:\n{content}\n")  # Debug output

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
        except Exception as e:
            state["error"] = f"Code generation error: {str(e)}"
        return state

    def _save_code(
        self,
        design_code: str,
        testbench_code: str,
        output_dir: str = "output",
    ) -> List[str]:
        """
        Save SystemVerilog code to files in a module-specific folder.

        Args:
          design_code: Generated design code.
          testbench_code: Generated testbench code.
          output_dir: Base directory for saving files (default: 'output').

        Returns:
          List[str]: Messages indicating save status or errors.
        """
        messages = []
        try:
            # Extract module name
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
            with open(design_filepath, "w") as f:
                f.write(design_code)
            messages.append(f"Design saved to {design_filepath}")

            # Save testbench code if available
            if testbench_code:
                testbench_filename = f"{module_name}_tb.sv"
                testbench_filepath = os.path.join(
                    module_dir, testbench_filename
                )
                with open(testbench_filepath, "w") as f:
                    f.write(testbench_code)
                messages.append(f"Testbench saved to {testbench_filepath}")
            else:
                messages.append("No testbench code to save")

            return messages
        except Exception as e:
            return [f"Error saving code: {str(e)}"]

    def generate(
        self,
        user_request: str,
        save_file: bool = False,
        output_dir: str = "output",
    ) -> Dict[str, Any]:
        """
        Generate SystemVerilog design and testbench.

        Args:
          user_request: Desired SystemVerilog module description.
          save_file: Save generated code to files (default: False).
          output_dir: Directory for saving files (default: 'output').

        Returns:
          Dict: Contains success, design_code, testbench_code, error, messages.
        """
        initial_state = {
            "user_request": user_request,
            "generated_code": "",
            "testbench_code": "",
            "messages": [HumanMessage(content=user_request)],
            "error": "",
        }
        result = self.graph.invoke(initial_state)

        # Save code if requested and no errors
        if save_file and not result["error"]:
            save_messages = self._save_code(
                result["generated_code"], result["testbench_code"], output_dir
            )
            for msg in save_messages:
                result["messages"].append(AIMessage(content=msg))

        return {
            "success": not bool(result["error"]),
            "design_code": result["generated_code"],
            "testbench_code": result["testbench_code"],
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
            print("Generated SystemVerilog Design:")
            print("-" * 40)
            print(result["design_code"])
            print("-" * 40)
            print("Generated SystemVerilog Testbench:")
            print("-" * 40)
            print(result["testbench_code"] or "No testbench generated")
            print("-" * 40)
            print("✓ Design and testbench generation successful")
        else:
            print(f"❌ Error: {result['error']}")

        # Show agent messages
        print("\nAgent Messages:")
        for msg in result["messages"]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            print(f"[{role}]: {msg.content}")

        print("\n" + "=" * 60)
