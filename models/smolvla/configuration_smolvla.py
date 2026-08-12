"""Local alias to the active LeRobot SmolVLA config class.

Using the environment class avoids duplicate draccus registration for the same
`smolvla` policy key when both local and site-packages modules are imported.
"""

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

__all__ = ["SmolVLAConfig"]
