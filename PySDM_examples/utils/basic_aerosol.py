from typing import Dict, Tuple

from PySDM.physics.constants_defaults import Mv, rho_w


class BasicAerosol:
    def __init__(
        self,
        *,
        densities: Dict[str, float],
        compounds: Tuple[str],
        molar_masses: Dict[str, float],
        is_soluble: Dict[str, bool],
        ionic_dissociation_phi: Dict[str, int]
    ):
        self._aerosol_modes = None
        self.densities = densities
        self.compounds = compounds
        self.molar_masses = molar_masses
        self.is_soluble = is_soluble
        self.ionic_dissociation_phi = ionic_dissociation_phi

    @property
    def aerosol_modes(self):
        return self._aerosol_modes

    @aerosol_modes.setter
    def aerosol_model(self, value: Tuple[Dict]):
        self._aerosol_modes = value

    # convert mass fractions to volume fractions
    def volume_fractions(self, mass_fractions: dict):
        return {
            k: (mass_fractions[k] / self.densities[k])
            / sum(mass_fractions[i] / self.densities[i] for i in self.compounds)
            for k in self.compounds
        }

    # calculate total volume fraction of soluble species
    def f_soluble_volume(self, mass_fractions: dict):
        volfrac = self.volume_fractions(mass_fractions)
        return sum(self.is_soluble[k] * volfrac[k] for k in self.compounds)

    # calculate volume fractions of just soluble or just insoluble species
    def volfrac_just_soluble(self, volfrac: dict, soluble=True):
        if soluble:
            _masked = {k: (self.is_soluble[k]) * volfrac[k] for k in self.compounds}
        else:
            _masked = {k: (not self.is_soluble[k]) * volfrac[k] for k in self.compounds}

        _denom = sum(list(_masked.values()))
        if _denom == 0.0:
            x = {k: 0.0 for k in self.compounds}
        else:
            x = {k: _masked[k] / _denom for k in self.compounds}
        return x

    # calculate hygroscopicities with different assumptions about solubility
    def kappa(self, mass_fractions: dict):
        volfrac = self.volume_fractions(mass_fractions)
        molar_volumes = {
            i: self.molar_masses[i] / self.densities[i] for i in self.compounds
        }

        result = {}
        for model in ("bulk", "film"):
            if model == "bulk":
                ns_per_vol = sum(
                    self.ionic_dissociation_phi[i] * volfrac[i] / molar_volumes[i]
                    for i in self.compounds
                )
            elif model == "film":
                volume_fractions_of_just_soluble = self.volfrac_just_soluble(
                    volfrac, soluble=True
                )
                ns_per_vol = self.f_soluble_volume(mass_fractions) * sum(
                    self.ionic_dissociation_phi[i]
                    * volume_fractions_of_just_soluble[i]
                    / molar_volumes[i]
                    for i in self.compounds
                )
            else:
                raise AssertionError()
            result[model] = ns_per_vol * Mv / rho_w
        return result

    # calculate molar volume of just organic species
    def nu_org(self, mass_fractions: dict):
        volfrac = self.volume_fractions(mass_fractions)
        molar_volumes = {
            i: self.molar_masses[i] / self.densities[i] for i in self.compounds
        }
        volume_fractions_of_just_org = self.volfrac_just_soluble(volfrac, soluble=False)
        return sum(
            volume_fractions_of_just_org[i] * molar_volumes[i] for i in self.compounds
        )
