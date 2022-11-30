import numpy as np
import torch
import dataclasses
import xml.etree.ElementTree as ET
import drjit as dr

def _nonzero(x, eta=1e-10):
    # x = x.clone()
    # x[x==0.] = eta

    return torch.where(x==0., eta, x)

def to_sphe_coords(dirs):
    '''
    Convert [n, 3] dirs into [n, 2] spherical coordinates
    '''

    theta = torch.arctan(dirs[:,1] / _nonzero(dirs[:, 0]))
    phi = torch.arctan(torch.sqrt(dirs[:, 0]**2 + dirs[:, 1]**2) / _nonzero(dirs[:, 2]))

    return torch.stack([theta, phi], axis=-1)

@dataclasses.dataclass
class BounceInfo():
    '''
    Class to log all bounce info for NRC training
    '''

    # 
    max_bounce: int = 0
    # 
    instant_radiance: dict = dataclasses.field(default_factory=lambda:{})
    # mi.Bool
    active: dict = dataclasses.field(default_factory=lambda:{})
    # 
    depth: dict = dataclasses.field(default_factory=lambda:{})
    # mi.Interaction3f
    si: dict = dataclasses.field(default_factory=lambda:{})
    # 
    bsdf_weight: dict = dataclasses.field(default_factory=lambda:{})
    # 
    nrc_results: dict = dataclasses.field(default_factory=lambda:{})
    # Diffuse reflectance in numpy array
    diffuse_reflectance: dict = dataclasses.field(default_factory=lambda:{})
    # Specular reflectance in mi.Color3f
    specular_reflectance: dict = dataclasses.field(default_factory=lambda:{})
    # Roughness reflectance in numpy array
    roughness: dict = dataclasses.field(default_factory=lambda:{})

    def log_data(
        self, 
        depth, 
        si, active, bsdf_weight, instant_radiance, nrc_results,
        diffuse_reflectance, specular_reflectance, roughness
        ):

        self.depth[self.max_bounce] = depth
        self.si[self.max_bounce] = si
        self.active[self.max_bounce] = active
        self.bsdf_weight[self.max_bounce] = bsdf_weight
        self.instant_radiance[self.max_bounce] = instant_radiance
        self.nrc_results[self.max_bounce] = nrc_results
        self.diffuse_reflectance[self.max_bounce] = diffuse_reflectance
        self.specular_reflectance[self.max_bounce] = specular_reflectance
        self.roughness[self.max_bounce] = roughness

        self.max_bounce += 1


    def gather_training_data(self):
        inputs = []
        outputs = []

        p = []
        wi = []
        n = []
        diff_ref = []
        spec_ref = []
        rough = []

        radiance = 0
        bounce_ids =[]
        for bounce_id in reversed(range(0, self.max_bounce)):
            # features = get_input_features_at_bounce(bounce_info, bounce_id, encode=encoded_inputs)
            # bounce_key = 'bounce_{}'.format(bounce_id)
            radiance_at_bounce = self.instant_radiance[bounce_id]
            bsdf_weight        = self.bsdf_weight[bounce_id]
            #nrc_prediction_at_bounce = bounce_info[bounce_key]['nrc']#[inds]
            active_mask = self.active[bounce_id] # TODO effie
            # Accumulate radiance backwards.
            radiance = radiance_at_bounce + bsdf_weight * radiance #+ (1-bsdf_weight) * nrc_prediction_at_bounce

            if len(active_mask)>1:
                # inputs.append(self.get_training_input(bounce_id))
                # gather inputs
                p.append(self.si[bounce_id].p.numpy()[active_mask])
                wi.append(self.si[bounce_id].wi.numpy()[active_mask])
                n.append(self.si[bounce_id].n.numpy()[active_mask])
                diff_ref.append(self.diffuse_reflectance[bounce_id][active_mask]) 
                spec_ref.append(self.specular_reflectance[bounce_id].numpy()[active_mask]) 
                rough.append(self.roughness[bounce_id][active_mask, None])

                outputs.append(radiance.numpy()[active_mask,:])
                bounce_ids.append(np.array([bounce_id]*dr.count(active_mask)))

        # Reshape so different bounces are just different samples
        p = np.concatenate(p,0)
        wi = np.concatenate(wi,0)
        n = np.concatenate(n,0)
        diff_ref = np.concatenate(diff_ref,0)
        spec_ref = np.concatenate(spec_ref,0)
        rough = np.concatenate(rough,0)

        outputs = np.concatenate(outputs,0)
        bounce_ids = np.concatenate(bounce_ids)
        print("All bounce_ids histogram {}".format(np.histogram(bounce_ids, [0,1,2,3,4])[0]))

        data = [[p, wi, n, diff_ref, spec_ref, rough], outputs]

        return data

    def get_training_input(self, bounce_id):
        mask = self.active[bounce_id]
        return [
            self.si[bounce_id].p.numpy()[mask],
            self.si[bounce_id].wi.numpy()[mask],
            self.si[bounce_id].n.numpy()[mask],
            self.diffuse_reflectance[bounce_id][mask], 
            self.specular_reflectance[bounce_id].numpy()[mask], 
            self.roughness[bounce_id][mask, None]
        ]




class ReflectanceInfo():
    def __init__(self, input_scene_path, ind_array, bsdf):
        # copy indices from GPU onto CPU
        ind_array = ind_array.numpy()

        # construct id to properties dict using:
        # (id (from reinterpret_array_v) to bsdf_str) and (bsdf_str to reflectances/roughness)
        self.str_prop_dict = {}
        self.id_str_dict = {}
        self.id_prop_dict = {}

        self.get_str_prop_dict(input_scene_path)
        self.get_id_prop_dict(ind_array, bsdf)

    def get_str_prop_dict(self, input_scene_path):
        """ 
        Method to return a dictionary of:
            key: BSDF id
            value: [diffuse_reflectance, specular_reflectance, roughness]
        """
        tree = ET.parse(input_scene_path)
        root = tree.getroot()

        for bsdf in root.iter('bsdf'):

            # give a default value for diffuse reflectance ([-1,-1,-1]), specular reflectance ([-1,-1,-1]), and roughness (-1)
            if bsdf.get('id') is not None:
                self.str_prop_dict[bsdf.attrib['id']] = [[-1,-1,-1], [-1,-1,-1], -1]

                # get reflectance and roughness
                diff_refl = ReflectanceInfo.to_list(ReflectanceInfo.get_value([-1,-1,-1], bsdf, 'name', ['reflectance', 'diffuseReflectance', 'diffuse_reflectance']))
                spec_refl = ReflectanceInfo.to_list(ReflectanceInfo.get_value([-1,-1,-1], bsdf, 'name', ['specularReflectance', 'specular_reflectance']))
                roughness = float(ReflectanceInfo.get_value(-1, bsdf, 'name', ['alpha']))

                # append to properties dictionary
                view = self.str_prop_dict[bsdf.attrib['id']]
                view[0], view[1], view[2] = diff_refl, spec_refl, roughness


    def get_id_prop_dict(self, ind_array, bsdf):

        l, i = np.unique(ind_array, return_index=True)
        # l: array of unique elements in ind_array, e.g. [4, 6, 8, etc.]
        # i: array of indices of first time appearances of each unique element in the list

        for idx, loc in zip(l, i):
            if idx not in self.id_str_dict:
                if bsdf.entry_(loc) is not None:
                    self.id_str_dict[idx] = bsdf.entry_(loc).id()
        
        # create a (127 x 7) matrix (for numpy parallelisation), default: -1
        # l[-1] is the largest element (for elements not present, e.g. 5, just kept as default)
        self.id_prop_dict = np.ones((l[-1] + 1, 7))
        self.id_prop_dict *= -1

        # populate the matrix using values in prop_dict, if not, keep default
        for m_id, m_str_name in self.id_str_dict.items():
            if m_str_name in self.str_prop_dict:
                v1, v2, v3 = self.str_prop_dict[m_str_name]
                self.id_prop_dict[m_id, :] = np.array([*v1, *v2, v3])
        # self.id_prop_dict = {k : self.prop_dict.get(v, default) for k, v in self.id_dict.items()}


    def get_properties(self, ind_array):

        def foo(val): 
            return self.id_prop_dict[val]
        
        #start = time.time()
        values = foo(ind_array)
        #print(f"map takes {time.time()- start}")
        
        return values[:, :3], values[:, 3:-1], values[:, -1]

    def get_value(default, ref, attribute, attribute_content):
        """ 
        Given inputs, return value associated with that element (that has a certain attribute name)
        Inputs:
            default: default value ([-1,-1,-1] for reflectance and -1 for roughness)
            ref: which root branch to base the search on
            attribute: (str) which attribute to search for, e.g. 'reflectance' or 'diffuseReflectance'
            attribute_content: [list] that the function will search through
        """
        for content in attribute_content:
            ele = ref.find(".//*[@%s='%s']"%(attribute, content))
            if ele is not None and ele.get('value') is not None:
                return ele.attrib['value']

        return default


    def to_list(value):
        """ 
        Method to return list of floats given input, if input is string
        """
        if isinstance(value, str):
            return [float(n) for n in value.split(',')]
        else:
            return value


# equation 4 in NRC paper: spread at primary vertex
def primary_vertex_spread_dr(x0, x1, wi, n, eta=1e-10):
    cos_theta = calculate_cos_dr(wi, n)
    a_0 = dr.sum((x0 - x1)**2) / dr.maximum(4 * np.pi * dr.abs(cos_theta), eta)
    return a_0

# cos(theta) between two arrays
def calculate_cos_dr(t1, t2, eta=1e-10):
    return dr.sum(t1 * t2) / dr.maximum(dr.norm(t1) * dr.norm(t2), eta)

# equation 3 in NRC paper: spread at n-th vertex (to be added into the summation)
def vertex_spread_dr(x_i_1, x_i, pdf_bsdf, wi, n):
    cos_theta = calculate_cos_dr(wi, n)
    nominator = dr.sum((x_i_1 - x_i)**2)
    denominator = pdf_bsdf * dr.abs(cos_theta)
    return dr.sqrt(nominator / denominator)