from __future__ import annotations

from vadbench.algorithms.base import VADAlgorithm
from vadbench.algorithms.neural import (
    AttnTCNLiteAlgorithm,
    CausalCRNNVADKWSAlgorithm,
    CausalCRNNVADMicroAlgorithm,
    CausalCRNNVADNanoAlgorithm,
    CausalCRNNVADTinyAlgorithm,
    CausalDSCNNGRUVADKWSAlgorithm,
    CNNTDLikeAlgorithm,
    CRNNVADAlgorithm,
    DSCNNVADKWSMatchAlgorithm,
    DSCNNVADLargeAlgorithm,
    DSCNNVADMediumAlgorithm,
    DSCNNVADSmallAlgorithm,
    DSCNNVADTinyAlgorithm,
    MarbleNet3x2x64Algorithm,
    MarbleNetLiteAlgorithm,
    SelfAttentiveVADAlgorithm,
    TinyMelCNNAlgorithm,
)
from vadbench.algorithms.traditional import (
    EnergyAdaptiveVAD,
    KaldiEnergyVAD,
    MFCCGMMVAD,
    RVADFastVAD,
    SohnHMMVAD,
    SpectralFluxLTSDVAD,
    SpectralGateVAD,
    WebRTCVAD,
    ZCREnergyVAD,
)


_ALGORITHMS: dict[str, type[VADAlgorithm]] = {
    EnergyAdaptiveVAD.name: EnergyAdaptiveVAD,
    ZCREnergyVAD.name: ZCREnergyVAD,
    SpectralGateVAD.name: SpectralGateVAD,
    MFCCGMMVAD.name: MFCCGMMVAD,
    KaldiEnergyVAD.name: KaldiEnergyVAD,
    SpectralFluxLTSDVAD.name: SpectralFluxLTSDVAD,
    SohnHMMVAD.name: SohnHMMVAD,
    RVADFastVAD.name: RVADFastVAD,
    WebRTCVAD.name: WebRTCVAD,
    TinyMelCNNAlgorithm.name: TinyMelCNNAlgorithm,
    MarbleNetLiteAlgorithm.name: MarbleNetLiteAlgorithm,
    AttnTCNLiteAlgorithm.name: AttnTCNLiteAlgorithm,
    MarbleNet3x2x64Algorithm.name: MarbleNet3x2x64Algorithm,
    CNNTDLikeAlgorithm.name: CNNTDLikeAlgorithm,
    CRNNVADAlgorithm.name: CRNNVADAlgorithm,
    CausalCRNNVADNanoAlgorithm.name: CausalCRNNVADNanoAlgorithm,
    CausalCRNNVADTinyAlgorithm.name: CausalCRNNVADTinyAlgorithm,
    CausalCRNNVADMicroAlgorithm.name: CausalCRNNVADMicroAlgorithm,
    CausalCRNNVADKWSAlgorithm.name: CausalCRNNVADKWSAlgorithm,
    CausalDSCNNGRUVADKWSAlgorithm.name: CausalDSCNNGRUVADKWSAlgorithm,
    SelfAttentiveVADAlgorithm.name: SelfAttentiveVADAlgorithm,
    DSCNNVADTinyAlgorithm.name: DSCNNVADTinyAlgorithm,
    DSCNNVADSmallAlgorithm.name: DSCNNVADSmallAlgorithm,
    DSCNNVADMediumAlgorithm.name: DSCNNVADMediumAlgorithm,
    DSCNNVADLargeAlgorithm.name: DSCNNVADLargeAlgorithm,
    DSCNNVADKWSMatchAlgorithm.name: DSCNNVADKWSMatchAlgorithm,
}


def create_algorithm(name: str, **kwargs: object) -> VADAlgorithm:
    try:
        cls = _ALGORITHMS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}") from exc
    return cls(**kwargs)


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHMS)
