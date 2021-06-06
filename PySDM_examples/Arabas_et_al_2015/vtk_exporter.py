import numpy as np
from evtk.hl import pointsToVTK


class VTKExporter:
    def __init__(self, particles_file_number=1, particles_file_path="./sd_points"):
        self.particles_file_number = particles_file_number
        self.particles_file_path = particles_file_path
        
    def export_particles(self, sd):
        path = self.particles_file_path + '_num' + str(self.particles_file_number)
        print("Exporting to vtk, path: " + path)
        self.particles_file_number += 1
            
        payload = {}
        
        for k in sd.attributes.keys():
            if len(sd[k].shape) != 1:
                tmp = sd[k].to_ndarray()
                tmp_dict = {k + '[' + str(i) + ']' : tmp[i] for i in range(len(sd[k].shape))}

                payload.update(tmp_dict)
            else:
                payload[k] = sd[k].to_ndarray()
  
        payload.update({k: np.array(v) for k, v in payload.items() if not (v.flags['C_CONTIGUOUS'] or v.flags['F_CONTIGUOUS'])})

        if len(sd['cell origin'].shape) == 2:
            x = payload['cell origin[0]'] + payload['position in cell[0]']
            y = np.full_like(x, 0)
            z = payload['cell origin[1]'] + payload['position in cell[1]']
        else:
            raise NotImplementedError("Only 2 dimensions array is supported at the moment.")

        pointsToVTK(path, x, y, z, data = payload) 
