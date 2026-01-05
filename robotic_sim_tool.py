
from typing import Dict, Any

from tooluniverse.tool_registry import register_tool
from tooluniverse.base_tool import BaseTool

from robotic_sim_core import run_robotic_simulation
from protocol_translator import translate_protocol_to_pylabrobot


@register_tool("RoboticSimulationTool", config={
    "name": "robotic_simulation",
    "description": (
        "Run a PyLabRobot based simulation of an experiment protocol and "
        "return executed commands, final deck state, and synthetic results."
    ),
    "parameter": {
        "type": "object",
        "properties": {
            "protocol_text": {
                "type": "string",
                "description": (
                    "Natural language description of the liquid handling "
                    "experiment protocol to simulate."
                )
            }
        },
        "required": ["protocol_text"]
    }
})
class RoboticSimulationTool(BaseTool):
    """
    ToolUniverse facing wrapper for your robotic simulation.

    ToolUniverse will call this class when the AI scientist chooses the
    'robotic_simulation' tool.
    """

    def run(self, arguments=None, **kwargs) -> Dict[str, Any]:
        """
        Expected call from ToolUniverse:

            tu.run({
                "name": "robotic_simulation",
                "arguments": {"protocol_text": "..."}
            })

        This method unpacks protocol_text, calls the core simulator,
        and returns a result dict with a 'success' flag.
        """

        # Support both ToolUniverse style (arguments dict) and direct calls
        if arguments is None:
            arguments = kwargs

        if isinstance(arguments, dict):
            protocol_text = arguments.get("protocol_text", "")
        else:
            # If someone passes a bare string, treat it as the protocol
            protocol_text = str(arguments)

        if not protocol_text:
            return {
                "success": False,
                "error": "protocol_text is required for robotic_simulation"
            }
        try:
            protocol_code = translate_protocol_to_pylabrobot(protocol_text)
            result = run_robotic_simulation(protocol_code)

        # Make sure there is a success flag for ToolUniverse
            result["success"] = True
            return result
        except Exception as e:
                return {
                    "success": False,
                    "error": f"Simulation failed: {e}"
                }
