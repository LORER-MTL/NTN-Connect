"""Channel post-processing components.

Each file implements one AbstractChannelComponent. The ChannelPipeline
picks which ones to apply based on cfg flags.
"""
from ntnconnect.registry.channels.components.shadowing import ShadowingComponent
from ntnconnect.registry.channels.components.doppler import DopplerComponent
from ntnconnect.registry.channels.components.propagation_delay import PropagationDelayComponent
from ntnconnect.registry.channels.components.polarization import PolarizationComponent
from ntnconnect.registry.channels.components.spatial_correlation import SpatialCorrelationComponent
from ntnconnect.registry.channels.components.jakes import JakesCorrelationComponent

__all__ = [
    "ShadowingComponent",
    "DopplerComponent",
    "PropagationDelayComponent",
    "PolarizationComponent",
    "SpatialCorrelationComponent",
    "JakesCorrelationComponent",
]
