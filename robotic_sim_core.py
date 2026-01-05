# robotic_sim_core.py

from typing import Dict, Any
import asyncio
import traceback

from pylabrobot.resources import Deck
from pylabrobot.liquid_handling import LiquidHandler
from protocol_translator import translate_protocol_to_pylabrobot as translation_LLM


async def _run_async_pylabrobot_sim(pylabrobot_code: str) -> Dict[str, Any]:
    """
    Runs the PyLabRobot code and returns a deck_state dict.
    """

    ns: Dict[str, Any] = {
        "Deck": Deck,
        "LiquidHandler": LiquidHandler,
    }

    exec(pylabrobot_code, ns)

    deck = None
    backend = None

    # Preferred: async run_protocol()
    run_protocol = ns.get("run_protocol")
    if run_protocol is not None:
        result = await run_protocol()
        if isinstance(result, tuple) and len(result) == 2:
            deck, backend = result
        elif isinstance(result, dict):
            deck = result.get("deck")
            backend = result.get("backend")


    if deck is None:
        deck = ns.get("deck")
    if backend is None:
        backend = ns.get("backend")

    if deck is None or backend is None:
        raise RuntimeError(
            "Generated PyLabRobot code did not define run_protocol() "
            "nor global deck/backend."
        )

    return serialize_deck_state(deck)


def run_pylabrobot_sim(pylabrobot_code: str) -> Dict[str, Any]:
    return asyncio.run(_run_async_pylabrobot_sim(pylabrobot_code))


def run_robotic_simulation(protocol_text: str) -> Dict[str, Any]:
    """
    Main function used by the agent.
    Now returns:
        { "success": True/False, "reason": "<string>" }
    """

    last_error = ""
    max_retries = 5

    for attempt in range(max_retries):

    # GIVE ERROR FEEDBACK TO TRANSLATOR
        if last_error:
            translator_input = (
                protocol_text
                + "\n\nThe previous PyLabRobot code failed with the following error:\n"
                + last_error
                + "\nFix your output so that this error does not occur. Regenerate the full PyLabRobot code."
            )
        else:
            translator_input = protocol_text

        # 1. Translate
        try:
            pylabrobot_code = translation_LLM(protocol_text)
        except Exception as e:
            last_error = f"Translator crashed: {e}"
            continue

        # 2. Simulate
        try:
            deck_state = run_pylabrobot_sim(pylabrobot_code)
        except Exception as e:
            last_error = f"Execution error: {e}"
            continue

        # 3. Validate
        valid, reason = validate_simulation(deck_state)
        if valid:
            return {
                "success": True,
                "reason": "Protocol executed successfully and passed validation.",
            }
        else:
        # if failed:  final verdict
            return {
                "success": False,
                "reason": f"Validation failed: {reason}",
            }


    return {"success": False, "reason": f"Failed after 5 retries. Last error: {last_error}"}


def serialize_deck_state(deck: Deck) -> Dict[str, Any]:
    """
    Manually extract volume state from all wells on the deck.
    """
    state = {}
    

    children = deck.children if hasattr(deck, 'children') else []
    

    if isinstance(children, dict):
        children_items = list(children.items())
    elif isinstance(children, list):
        children_items = [(getattr(r, 'name', f'resource_{i}'), r) for i, r in enumerate(children)]
    else:
        children_items = []
    
 
    for resource_name, resource in children_items:
        if hasattr(resource, 'children'):  # It's a carrier
            carrier_children = resource.children
            
            
            if isinstance(carrier_children, dict):
                carrier_items = list(carrier_children.items())
            elif isinstance(carrier_children, list):
                carrier_items = [(getattr(r, 'name', f'child_{i}'), r) for i, r in enumerate(carrier_children)]
            else:
                carrier_items = []
            
            for child_name, child_resource in carrier_items:
                # Check if this is a holder that contains another resource (the actual plate)
                if hasattr(child_resource, 'children') and child_resource.children:
                    holder_children = child_resource.children
                    
                    # The holder might have a plate inside
                    if isinstance(holder_children, list) and len(holder_children) > 0:
                        actual_plate = holder_children[0]  # Get the first (usually only) child
                        
                        # Now check if this actual plate has wells
                        if hasattr(actual_plate, '__getitem__'):
                            wells_state = {}
                            try:
                                # Try common well positions to find all wells with liquid
                                rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                                cols = range(1, 13)  # 1-12
                                
                                for row in rows:
                                    for col in cols:
                                        well_id = f"{row}{col}"
                                        try:
                                            well = actual_plate[well_id][0]
                                            if hasattr(well, 'tracker'):
                                                volume = well.tracker.get_used_volume()
                                                max_volume = well.max_volume if hasattr(well, 'max_volume') else None
                                                
                                                wells_state[well_id] = {
                                                    'total_volume_ul': volume,
                                                    'max_volume_ul': max_volume
                                                }
                                        except (KeyError, IndexError, AttributeError):
                                            # Well doesn't exist or can't be accessed
                                            pass
                                
                                if wells_state:
                                    plate_name = actual_plate.name if hasattr(actual_plate, 'name') else child_name
                                    state[plate_name] = {'wells': wells_state}
                            except Exception as e:
                                pass
    
    return {
        "structure": deck.serialize() if hasattr(deck, 'serialize') else None,
        "state": state,
    }

def validate_simulation(deck_state: Dict[str, Any]) -> (bool, str):
    """
    Validation decides if the simulation result looks physically valid.

    Returns:
        (True,  "ok")
        (False, "reason")
    """

    state = deck_state.get("state", {}) or {}

    all_vols = []

    # Track negative and overflow checks
    for res_name, res_state in state.items():
        wells = res_state.get("wells", {})
        for well_name, w in wells.items():

            vol = w.get("total_volume_ul", 0.0)
            all_vols.append(vol)

            if vol < 0:
                return False, f"Negative volume in {res_name}:{well_name}"

            max_vol = w.get("max_volume_ul")
            if max_vol is not None and vol > max_vol:
                return False, f"Overflow in {res_name}:{well_name} (vol={vol}, max={max_vol})"

    # If no volume anywhere, nothing happened
    if not all_vols or all(v == 0 for v in all_vols):
        return False, "No liquid moved (all wells 0 µL)."

    # If we reach here, everything is OK
    return True, "ok"
