"""
Created at 25.09.2019
"""

from PySDM_examples.Arabas_et_al_2015.netcdf_exporter import NetCDFExporter
from PySDM_examples.Arabas_et_al_2015.settings import Settings, si
from PySDM_examples.Arabas_et_al_2015.simulation import Simulation
from PySDM_examples.Arabas_et_al_2015.storage import Storage
from PySDM_examples.utils.temporary_file import TemporaryFile

from PySDM_examples.Arabas_et_al_2015.vtk_exporter import VTKExporter

def main():
    settings = Settings()

    settings.n_sd_per_gridbox = 25
    settings.grid = (25, 25)
    settings.simulation_time = 5400 * si.second

    storage = Storage()
    simulation = Simulation(settings, storage)
    simulation.reinit()
    
    exporter = VTKExporter()

    for step in range(settings.n_steps):
        simulation.core.run(1)

        exporter.export_particles(simulation.core.particles)

    #simulation.run()
    #temp_file = TemporaryFile('.nc')
    #exporter = NetCDFExporter(storage, settings, simulation, temp_file.absolute_path)
    #exporter.run()



if __name__ == '__main__':
    main()    
