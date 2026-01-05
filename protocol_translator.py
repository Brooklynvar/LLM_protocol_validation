# protocol_translator.py

from typing import List, Tuple
from openai import OpenAI
from pylabrobot_rag_tool import PyLabRobotRAGTool


rag = PyLabRobotRAGTool(
    json_path="pylabrobot_capabilities_detailed.json",
    markdown_path="pylabrobot_capabilities.md",
    db_path="./lancedb_db",
    verbose=False
)

client = OpenAI()

SYSTEM_PROMPT = """
You are an assistant that translates natural-language liquid-handling protocols
into executable Python code using PyLabRobot.

Your output must be VALID PYTHON CODE ONLY. No explanations, no markdown.
Comments (#) are allowed but optional.

You must generate a FULL PROTOCOL that defines:

    async def run_protocol():
        ...
        return deck, backend

You are a PyLabRobot code generator. Convert the user’s protocol description into a full,
executable PyLabRobot script. Follow ALL rules below without exception.

────────────────────────────────────────────────────────────
 GLOBAL REQUIREMENTS
────────────────────────────────────────────────────────────

- Always output ONLY Python code (no explanations).
- Always add `import asyncio` at the top.
- Import exactly the PyLabRobot classes/libraries needed:
    Decks, resources, Hamilton carriers, backends, LiquidHandler,
    and helper functions (like set_tip_tracking if used).
- Use `LiquidHandlerChatterboxBackend` unless the user requests a specific backend.
- Create a LiquidHandler with backend + deck, then call:
        await lh.setup()

- Place all labware using:
        deck.assign_child_resource(carrier, rails=N)

- Use only resources listed in the PyLabRobot reference.
  Do NOT invent new classes or resources.
- If the user mentions unavailable hardware, leave a comment and substitute
  with a compatible PyLabRobot resource.

- NEVER call `asyncio.run()`.
- At the end of the script, return:
        return deck, backend

────────────────────────────────────────────────────────────
 CRITICAL VOLUME RULE  (MUST ALWAYS FOLLOW)
────────────────────────────────────────────────────────────

Before ANY aspirate(), you MUST explicitly set initial volume for that well using :

source_well.set_liquids([(None, 300.0)])
after defining : source_well = reservoir["A1"][0]

This applies to:
- every reservoir well you aspirate from
- every plate well you aspirate from
- every plate well you mix (because mixing uses aspirate→dispense cycles)
- every step of serial dilution

If the protocol does not specify a starting volume:
→ assume 300 µL by default.

NEVER aspirate from a well that has not been initialized.

────────────────────────────────────────────────────────────
 LIQUID HANDLING RULES
────────────────────────────────────────────────────────────

- pick_up_tips() MUST precede aspirate() or dispense().
- drop/return tips when appropriate.
- NEVER dispense unless you have first aspirated the same volume.
- ALL lh methods must receive lists:
      await lh.aspirate([well], vols=[50.0])

- For serial dilutions:
    • mix the source before transferring
    • aspirate from source → dispense to destination
    • always respect volumes

────────────────────────────────────────────────────────────
 PLATE & WELL RULES
────────────────────────────────────────────────────────────

- 96-well plates follow A1–H12.
- Access wells using:
        plate["A1"][0]

- Carriers such as `PLT_CAR_L5AC_A00` and `TIP_CAR_480_A00` must be imported from:
        pylabrobot.resources.hamilton

───────────────────────────────────────────────────────────────
 CRITICAL VOLUME TRACKING (CHATTERBOX BACKEND)
───────────────────────────────────────────────────────────────

The Chatterbox backend does NOT automatically track volumes.
You MUST manually update the volume tracker after EVERY aspirate and dispense:

After aspirate:
    await lh.aspirate([source_well], vols=[80.0])
    source_well.tracker.remove_liquid(80.0)

After dispense:
    await lh.dispense([dest_well], vols=[40.0])
    dest_well.tracker.add_liquid(None, 40.0)

For multi-well operations with lists:
    await lh.aspirate(source_wells, vols=[50.0] * 8)
    for well, vol in zip(source_wells, [50.0] * 8):
        well.tracker.remove_liquid(vol)

This is MANDATORY for validation to work correctly.

────────────────────────────────────────────────────────────
 REQUIRED SCRIPT STRUCTURE (use this pattern)
────────────────────────────────────────────────────────────

Your generated script MUST follow this exact structure:

    import asyncio
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.chatterbox import LiquidHandlerChatterboxBackend
from pylabrobot.resources.hamilton import STARLetDeck, PLT_CAR_L5AC_A00, TIP_CAR_480_A00
from pylabrobot.resources import (
    Cor_96_wellplate_360ul_Fb,
    hamilton_96_tiprack_1000uL_filter,
)


async def run_protocol():
    deck = STARLetDeck()
    backend = LiquidHandlerChatterboxBackend(num_channels=1)
    lh = LiquidHandler(backend=backend, deck=deck)
    await lh.setup()
    lh.update_volume_state = True

    # Plate carrier with reservoir and destination plate
    plate_carrier = PLT_CAR_L5AC_A00(name="plate_carrier")
    plate_carrier[0] = reservoir = Cor_96_wellplate_360ul_Fb(name="reservoir")
    plate_carrier[1] = dest_plate = Cor_96_wellplate_360ul_Fb(name="dest_plate")
    deck.assign_child_resource(plate_carrier, rails=15)

    # Tip carrier with one tip rack
    tip_carrier = TIP_CAR_480_A00(name="tip_carrier")
    tip_carrier[0] = tip_rack = hamilton_96_tiprack_1000uL_filter(name="tip_rack")
    deck.assign_child_resource(tip_carrier, rails=3)


    # Source and destination wells
    source_well = reservoir["A1"][0]
    dest_well_1 = dest_plate["B1"][0]
    dest_well_2 = dest_plate["B2"][0]

    #Initiliazation of well you are aspiring from
    source_well.set_liquids([(None, 300.0)])

    # Transfer 80 µL from reservoir A1, dispense 40 µL into B1 and 40 µL into B2
    await lh.pick_up_tips(tip_rack["A1"])
    await lh.aspirate([source_well], vols=[80.0])
    source_well.tracker.remove_liquid(80.0)
    await lh.dispense([dest_well_1], vols=[40.0])
    dest_well_1.tracker.add_liquid(None, 40.0) 
    await lh.dispense([dest_well_2], vols=[40.0])
    dest_well_2.tracker.add_liquid(None, 40.0)
    await lh.return_tips()


    await lh.stop()
    return deck, backend


"""


_FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = [
    (
        # EXAMPLE 1: Simple serial dilution on one 96 well plate
        "On a STARLet deck, use one 96 well plate on a plate carrier and one 1000 µL tiprack "
        "on a tip carrier. Fill columns 2 to 12 with 50 µL buffer, then perform a serial "
        "dilution across columns 1 to 12 by transferring 50 µL from each column to the next "
        "with an 8 channel head and mixing 3 times in each destination column.",
        '''\
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.chatterbox import LiquidHandlerChatterboxBackend
from pylabrobot.resources.hamilton import STARLetDeck, PLT_CAR_L5AC_A00, TIP_CAR_480_A00
from pylabrobot.resources import Cor_96_wellplate_360ul_Fb, hamilton_96_tiprack_1000uL_filter

async def run_protocol():
    deck = STARLetDeck()
    backend = LiquidHandlerChatterboxBackend(num_channels=8)
    lh = LiquidHandler(backend=backend, deck=deck)
    await lh.setup()

    # Plate carrier and plate
    plt_carrier = PLT_CAR_L5AC_A00(name="plate_carrier")
    plt_carrier[0] = plate = Cor_96_wellplate_360ul_Fb(name="plate")
    deck.assign_child_resource(plt_carrier, rails=15)

    # Tip carrier and tip rack
    tip_carrier = TIP_CAR_480_A00(name="tip_carrier")
    tip_carrier[0] = tip_rack = hamilton_96_tiprack_1000uL_filter(name="tip_rack")
    deck.assign_child_resource(tip_carrier, rails=3)

    # Add 50 µL buffer to columns 2–12
    for col in range(2, 13):
        wells = [plate[f"{row}{col}"][0] for row in "ABCDEFGH"]
        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.aspirate(wells, vols=[50.0] * 8)  # assume buffer already in these wells
        await lh.dispense(wells, vols=[50.0] * 8)
        await lh.return_tips()

    # Serial dilution across columns 1–12
    for col in range(1, 12):
        source_wells = [plate[f"{row}{col}"][0] for row in "ABCDEFGH"]
        dest_wells   = [plate[f"{row}{col+1}"][0] for row in "ABCDEFGH"]

        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.aspirate(source_wells, vols=[50.0] * 8)
        await lh.dispense(dest_wells, vols=[50.0] * 8)

        # Mix 3 times in destination column
        for _ in range(3):
            await lh.aspirate(dest_wells, vols=[50.0] * 8)
            await lh.dispense(dest_wells, vols=[50.0] * 8)

        await lh.return_tips()

    await lh.stop()
    return deck, backend
'''
    ),
    (
        # EXAMPLE 2: Plate replication
        "Copy the entire contents of a source 96 well plate to a destination 96 well plate "
        "on a STARLet deck. Use an 8 channel pipette to transfer 100 µL from each well in "
        "the source plate to the corresponding well in the destination plate, column by column.",
        '''\
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.chatterbox import LiquidHandlerChatterboxBackend
from pylabrobot.resources.hamilton import STARLetDeck, PLT_CAR_L5AC_A00, TIP_CAR_480_A00
from pylabrobot.resources import Cor_96_wellplate_360ul_Fb, hamilton_96_tiprack_1000uL_filter

async def run_protocol():
    deck = STARLetDeck()
    backend = LiquidHandlerChatterboxBackend(num_channels=8)
    lh = LiquidHandler(backend=backend, deck=deck)
    await lh.setup()

    # Plate carrier with source and destination plates
    plate_carrier = PLT_CAR_L5AC_A00(name="plate_carrier")
    plate_carrier[0] = source_plate = Cor_96_wellplate_360ul_Fb(name="source")
    plate_carrier[1] = dest_plate   = Cor_96_wellplate_360ul_Fb(name="destination")
    deck.assign_child_resource(plate_carrier, rails=15)

    # Tip carrier with one tip rack
    tip_carrier = TIP_CAR_480_A00(name="tip_carrier")
    tip_carrier[0] = tip_rack = hamilton_96_tiprack_1000uL_filter(name="tip_rack")
    deck.assign_child_resource(tip_carrier, rails=3)

    # Column wise replication with 8 channel head
    for col in range(1, 13):
        source_wells = [source_plate[f"{row}{col}"][0] for row in "ABCDEFGH"]
        dest_wells   = [dest_plate[f"{row}{col}"][0]   for row in "ABCDEFGH"]

        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.aspirate(source_wells, vols=[100.0] * 8)
        await lh.dispense(dest_wells,   vols=[100.0] * 8)
        await lh.return_tips()

    await lh.stop()
    return deck, backend
'''
    ),
    (
        # EXAMPLE 3: Cherry picking
        "On a STARLet deck with a source and destination 96 well plate, perform cherry picking: "
        "transfer 50 µL from source wells A1, B3, D5, F7, H9 to destination wells A1, A2, A3, A4, A5 "
        "respectively. Use a fresh tip for each transfer and discard tips after each move.",
        '''\
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.chatterbox import LiquidHandlerChatterboxBackend
from pylabrobot.resources.hamilton import STARLetDeck, PLT_CAR_L5AC_A00, TIP_CAR_480_A00
from pylabrobot.resources import Cor_96_wellplate_360ul_Fb, hamilton_96_tiprack_1000uL_filter

async def run_protocol():
    deck = STARLetDeck()
    backend = LiquidHandlerChatterboxBackend(num_channels=1)
    lh = LiquidHandler(backend=backend, deck=deck)
    await lh.setup()

    # Plate carrier with source and destination plates
    plate_carrier = PLT_CAR_L5AC_A00(name="plate_carrier")
    plate_carrier[0] = source_plate = Cor_96_wellplate_360ul_Fb(name="source")
    plate_carrier[1] = dest_plate   = Cor_96_wellplate_360ul_Fb(name="destination")
    deck.assign_child_resource(plate_carrier, rails=15)

    # Tip carrier with one tip rack
    tip_carrier = TIP_CAR_480_A00(name="tip_carrier")
    tip_carrier[0] = tip_rack = hamilton_96_tiprack_1000uL_filter(name="tip_rack")
    deck.assign_child_resource(tip_carrier, rails=3)

    # Cherry pick map
    transfer_map = [
        ("A1", "A1"),
        ("B3", "A2"),
        ("D5", "A3"),
        ("F7", "A4"),
        ("H9", "A5"),
    ]

    for source_name, dest_name in transfer_map:
        source_well = source_plate[source_name][0]
        dest_well   = dest_plate[dest_name][0]

        await lh.pick_up_tips(tip_rack["A1"])
        await lh.aspirate([source_well], vols=[50.0])
        await lh.dispense([dest_well],   vols=[50.0])

        # Drop tips to trash area
        await lh.drop_tips([deck.get_trash_area()])

    await lh.stop()
    return deck, backend
'''
    ),
    (
        # EXAMPLE 4: Multi step enzyme assay
        "Prepare a 96 well plate for an enzyme assay. Step 1: add 50 µL substrate to all wells. "
        "Step 2: add 25 µL enzyme from column 1 of a source plate to all assay wells in columns 1 to 12. "
        "Step 3: mix each well 5 times with 50 µL. Step 4: wait 30 seconds. "
        "Step 5: add 25 µL stop solution from a third plate to all wells.",
        '''\
import asyncio
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.chatterbox import LiquidHandlerChatterboxBackend
from pylabrobot.resources.hamilton import STARLetDeck, PLT_CAR_L5AC_A00, TIP_CAR_480_A00
from pylabrobot.resources import (
    Cor_96_wellplate_360ul_Fb,
    hamilton_96_tiprack_1000uL_filter,
)

async def run_protocol():
    deck = STARLetDeck()
    backend = LiquidHandlerChatterboxBackend(num_channels=8)
    lh = LiquidHandler(backend=backend, deck=deck)
    await lh.setup()

    # Plate carrier with substrate, enzyme source and assay plate
    plate_carrier = PLT_CAR_L5AC_A00(name="plate_carrier")
    plate_carrier[0] = substrate_plate = Cor_96_wellplate_360ul_Fb(name="substrate_plate")
    plate_carrier[1] = source_plate    = Cor_96_wellplate_360ul_Fb(name="enzyme_source")
    plate_carrier[2] = assay_plate     = Cor_96_wellplate_360ul_Fb(name="assay_plate")
    deck.assign_child_resource(plate_carrier, rails=15)

    # Tip carrier with one tip rack
    tip_carrier = TIP_CAR_480_A00(name="tip_carrier")
    tip_carrier[0] = tip_rack = hamilton_96_tiprack_1000uL_filter(name="tip_rack")
    deck.assign_child_resource(tip_carrier, rails=3)

    # Step 1: 50 µL substrate from substrate plate to all assay wells
    for col in range(1, 13):
        substrate_wells = [substrate_plate[f"{row}{col}"][0] for row in "ABCDEFGH"]
        dest_wells      = [assay_plate[f"{row}{col}"][0]     for row in "ABCDEFGH"]

        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.aspirate(substrate_wells, vols=[50.0] * 8)
        await lh.dispense(dest_wells,      vols=[50.0] * 8)
        await lh.return_tips()

    # Step 2: 25 µL enzyme from column 1 of source plate to all assay wells
    enzyme_wells = [source_plate[f"{row}1"][0] for row in "ABCDEFGH"]
    for col in range(1, 13):
        dest_wells = [assay_plate[f"{row}{col}"][0] for row in "ABCDEFGH"]
        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.aspirate(enzyme_wells, vols=[25.0] * 8)
        await lh.dispense(dest_wells,   vols=[25.0] * 8)
        await lh.return_tips()

    # Step 3: mix each well 5 times with 50 µL
    for col in range(1, 13):
        wells = [assay_plate[f"{row}{col}"][0] for row in "ABCDEFGH"]
        await lh.pick_up_tips(tip_rack["A1:H1"])
        for _ in range(5):
            await lh.aspirate(wells, vols=[50.0] * 8)
            await lh.dispense(wells, vols=[50.0] * 8)
        await lh.return_tips()

    # Step 4: incubation
    await asyncio.sleep(30)

    # Step 5: 25 µL stop solution from substrate_plate column 1 to all wells
    stop_wells = [substrate_plate[f"{row}1"][0] for row in "ABCDEFGH"]
    for col in range(1, 13):
        dest_wells = [assay_plate[f"{row}{col}"][0] for row in "ABCDEFGH"]
        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.aspirate(stop_wells, vols=[25.0] * 8)
        await lh.dispense(dest_wells, vols=[25.0] * 8)
        await lh.return_tips()

    await lh.stop()
    return deck, backend
'''
    )
]


def translate_protocol_to_pylabrobot(protocol_text: str) -> str:
    """
    Convert natural language to PyLabRobot Python code using RAG plus LLM.
    """

    # 1. Retrieve RAG context specifically relevant to this protocol
    rag_docs = rag.query(protocol_text, k=10)
    rag_context = "\n\n".join(rag_docs)

    # 2. Build messages with system prompt and RAG injected as its own system message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if rag_context:
        messages.append(
            {
                "role": "system",
                "content": "# PyLabRobot Reference:\n" + rag_context,
            }
        )

    # 3. Add few shot examples
    for nl, code in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": nl})
        messages.append({"role": "assistant", "content": code})

    # 4. Add the actual user protocol
    messages.append({"role": "user", "content": protocol_text})

    # 5. Call the model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    code = response.choices[0].message.content or ""

    print("\n================ TRANSLATED CODE ================")
    print(code)
    print("=================================================\n")

    return code.strip()


