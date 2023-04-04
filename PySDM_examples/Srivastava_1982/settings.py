from PySDM.physics import constants_defaults, si

DUMMY_FRAG_MASS = -1


class SimProd:
    def __init__(self, name, plot_title=None, plot_xscale=None, plot_yscale=None):
        self.name = name
        self.plot_title = plot_title or name
        self.plot_yscale = plot_yscale
        self.plot_xscale = plot_xscale


class SimProducts:
    class PySDM:
        total_numer = SimProd(
            name="total numer",
            plot_title="total droplet numer",
            plot_xscale="log",
            plot_yscale="log",
        )
        total_volume = SimProd(name="total volume")
        super_particle_count = SimProd(
            name="super-particle count", plot_xscale="log", plot_yscale="log"
        )

    class Computed:
        mean_drop_volume_total_volume_ratio = SimProd(
            name="mean drop volume / total volume %",
            plot_title="mean drop mass / total mass %",
        )

    def get_prod_by_name(name):
        for class_obj in (SimProducts.PySDM, SimProducts.Computed):
            for attribute_str in dir(class_obj):
                if not attribute_str.startswith("__"):
                    attribute = getattr(class_obj, attribute_str)
                    if attribute.name == name:
                        return attribute
        return None


class Settings:
    """interprets parameters from Srivastava 1982 in PySDM context"""

    def __init__(
        self,
        *,
        n_sds,
        srivastava_c,
        srivastava_beta=None,
        frag_mass=DUMMY_FRAG_MASS,
        # TODO: get rid of the surplus parameters below
        dt=1 * si.s,
        dv=1 * si.m**3,
        drop_mass_0=1 * si.g,
        rho=constants_defaults.rho_w,
        total_number=1e6
    ):
        self.rho = rho
        self.total_number_0 = total_number
        self.total_volume = self.total_number_0 * drop_mass_0 / self.rho
        self.dt = dt
        self.dv = dv
        self.frag_mass = frag_mass

        self.prods = (
            SimProducts.PySDM.total_volume.name,
            SimProducts.PySDM.total_numer.name,
            SimProducts.PySDM.super_particle_count.name,
        )
        self.n_sds = n_sds

        # TODO
        self.c = srivastava_c
        self.beta = srivastava_beta
