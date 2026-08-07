"""One-shot Isaac Sim migration for all hand-gripper preset geometry."""

import asyncio
import traceback

import omni.kit.app
import omni.usd
from omni.ext._impl import _internal


async def _run():
    return_code = 0
    try:
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        extension = None
        for ext_id, modules in _internal._extensions.items():
            if ext_id.startswith("hand_grip_making_tool-") and modules._started_extensions:
                extension = modules._started_extensions[0][0]
                break
        if extension is None:
            raise RuntimeError("Hand Grip Preset Maker extension instance was not found.")
        extension._load_gripper()
        await omni.kit.app.get_app().next_update_async()
        preferred = extension._preset_name_model.get_value_as_string()
        updated, skipped = await extension._recompute_all_preset_geometry_async(preferred)
        print(f"HAND_GRIP_RECOMPUTE: updated={updated}, skipped={skipped}")
    except Exception:
        return_code = 2
        traceback.print_exc()
        print("HAND_GRIP_RECOMPUTE: FAIL")
    finally:
        omni.kit.app.get_app().post_quit(return_code)


asyncio.ensure_future(_run())
