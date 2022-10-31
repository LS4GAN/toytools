"""
Post process simulated data Haiwang produced
To-do list:
    -
"""
import re
import sys
from itertools import product
from pathlib import Path
from multiprocessing import Pool
import yaml
import numpy as np
import fire


def print_dict(mydict, level=0):
    """
    Print dictionary recursively.
    """
    indentation = '\t' * level
    for key, val in mydict.items():
        if isinstance(val, dict):
            print(f'{indentation}{key}:')
            print_dict(val, level=level + 1)
        else:
            print(f'{indentation}{key}: {val}')


def grids(field_map):
    """
    Generate grid from a map
    Input:
        - field_map: a dictionary:
            example = {'param1': [3, 4],
                       'param2': ['dog', 'cat']}
    Output:
        list of parameter map:
            example = [{'param1': 3, 'param2': 'dog'},
                       {'param1': 3, 'param2': 'cat'},
                       {'param1': 4, 'param2': 'dog'},
                       {'param1': 4, 'param2': 'cat'}]
    """
    keys = field_map.keys()
    vals = field_map.values()
    for value_tuple in product(*vals):
        yield dict(zip(keys, value_tuple))


class PostProcess:
    """
    A more configurable post processer.
    Input and output file name/path are specified by a template,
    so we might not need to change the code every time.

    Hard-coded assumptions:
        There are still several assumptions that are hard-coded.
        Search "ASSUMPTION" to see the exact places where they are used.
        The hard-coded assumptions are listed below:
        - The job output folder are structured as JobID[_part]
            with optional "_part";
        - We assume that sub-files of a input file are indexed by
            keys with format [other stuff]_[event id] so
            we can parse the event id from the key.
        - We assume that each sub-file is 2-d matrix with rows
            indexed by wires in three planes:
            U [: 800], V [800: 1600], W [1600:].

    Sample config file: {

        # Location of the data.
        'input_folder': '/data/datasets/LS4GAN_yuhw/',

        # Job id prefix(es).
        # Each job output folder is prefixed by (one of) the job id.
        # The full job id may contain a part id, eg. 61503165_1.
        # Can be single string or a list of strings
        # ASSUMPTION: We do assume that the part id is a suffix to the
        # job id and separated from job id by an underscore.
        'JobID': '61503165',

        # Some part of the job output might be incomplete
        # and we can exclude them.
        # Can be a single string or a list of strings.
        # If nothing to be exclude, the field can be omitted.
        'exclude': '61503165_6',

        # Template of input filenames in the job output folder.
        # Each fields is contained in a bracket.
        # The fields specified in the input_template must match
        # the keys in the fields item below.
        'input_template': 'frames[Sim]-[SigProc][APA]_*.npz',

        # The values for each field specified in the input template.
        'fields': {
            'Sim': ('1d', '2d'),
            'SigProc': ('orig', 'gauss'),
            'APA': (0, 1, 2, 3, 4, 5)
        },

        # Location to save the output.
        'output_folder': '/data/datasets/LS4GAN_yuhw/full_images',

        # The path template of each output file.
        # The "Event" and "Plane" (U, V, W) are parsed from each input field.
        'output_template': '[SigProc]/[Sim]/JobID_[JobID]-Event_[Event]-[APA]-[Plane].npz'
    }

    To-does:
    """
    def __init__(self, config):

        self.config = config

        self.input_template = config['input_template']
        self.fields = config['fields']

        self.__verify()

        # job id prefix(es) to load
        if isinstance(config['JobID'], str):
            self.job_ids = (config['JobID'],)
        else:
            self.job_ids = tuple(config['JobID'])

        # process folders to exclude
        self.excludes = ()
        if 'exclude' in config:
            if isinstance(config['exclude'], str):
                self.excludes = (config['exclude'],)
            else:
                self.excludes = tuple(config['exclude'])

        # set up output
        self.output_folder = Path(config['output_folder'])
        if self.output_folder.exists():
            print(f"{config['output_folder']} is not empty.")
            sys.exit("Refuse to overwrite an exist folder")
        self.output_folder.mkdir(parents=True)
        self.output_template = config['output_template']


    def __verify(self):
        """
        Verify that values for all fields in the input template
        has been provided in the config.
        """
        input_keys = re.findall(r"\[(\w+)\]", self.config['input_template'])
        assert set(input_keys) == self.config['fields'].keys()


    def __load_folders(self):
        """
        Load all job output folders prefixed by given job id(s)
        from the input folder
        """
        # make sure the input folder exists
        input_folder = Path(self.config['input_folder'])
        assert input_folder.exists(), \
            f"input folder {input_folder} doesn't exist"

        # Locate all the job output folders
        # that are not in exclusion list.
        folders = {}
        for job_id_glob in self.job_ids:
            for folder in input_folder.glob(f'{job_id_glob}*'):
                job_id = folder.stem
                if job_id not in self.excludes:
                    folders[job_id] = folder

        return folders


    def __load_fname(self, folder):
        """
        From one job output folder,
        load all file names with fields in input_template
        replaced by a value from in the fields map.
        """
        result = []
        for parameter in grids(self.config['fields']):
            fname_glob = self.config['input_template']

            for key, val in parameter.items():
                fname_glob = fname_glob.replace(f'[{key}]', str(val))

            fnames = list(folder.glob(fname_glob))
            assert len(fnames) <= 1, \
                f'Multiple files might satisfies \
                  the fname template {fname_glob}'
            if len(fnames) == 1:
                # we have to return the parameters together
                # with the filename, because the parameters
                # are needed to construct the output path.
                for key, val in parameter.items():
                    if isinstance(self.config['fields'][key], dict):
                        parameter[key] = self.config['fields'][key][val]
                result.append([parameter, fnames[0]])

        return result


    def __load_fnames(self):
        """
        Load all fnames from all the job output folders.
        Also add the job id field to the parameters.
        """
        folders = self.__load_folders()
        result = []
        for job_id, folder in folders.items():
            for parameter_map, fname in self.__load_fname(folder):
                parameter_map['JobID'] = job_id
                result.append([parameter_map, fname])

        assert len(result) > 0, \
            'No data founded! Check job id(s), \
             input_template, and the field values.'

        return result


    @staticmethod
    def __load_npz(fname):
        """
        Load npz file containing multiple sub-files.
        ASSUMPTIONS:
        - We assume that sub-files of a input file are indexed by
            keys with format [other stuff]_[event id] so
            we can parse the event id from the key.
        - We assume that each sub-file is 2-d matrix with rows
            indexed by wires in three planes:
            U [: 800], V [800: 1600], W [1600:].
        """
        result = []
        with np.load(fname) as file_handle:
            frames = [key for key in file_handle.files if key.startswith('frame')]
            for frame in frames:
                event = frame.split('_')[-1]
                data = file_handle[frame]
                data_map = {'U': data[ : 800],
                            'V': data[800 : 1600],
                            'W': data[1600 : ]}
                for plane in data_map:
                    parameter = {'Event': event, 'Plane': plane}
                    result.append([parameter, data_map[plane]])

        return result


    def __save(self, parameter, data):
        """
        Save one full-plane image with a save path formulated with parameter.
        """
        # get save path
        save_path = self.output_template
        for key, val in parameter.items():
            save_path = save_path.replace(f'[{key}]', str(val))
        save_path = self.output_folder/save_path

        # make sure folder exists
        folder = save_path.parent
        if not folder.exists():
            folder.mkdir(parents=True)

        np.savez_compressed(save_path, data=data)


    # The following two functions is for processing files in parallel
    def work_log(self, item):
        """
        Load one output file and save all sub-files in it.
        """
        parameter, fname = item
        print(parameter)
        for param, data in self.__load_npz(fname):
            param.update(parameter)
            self.__save(param, data)


    def process(self, num_workers=4):
        """
        Process output files in parallel
        with a given number of workers
        """
        tasks = self.__load_fnames()
        pool = Pool(num_workers)
        pool.map(self.work_log, tasks)


def main(config_fname='./config.yaml',
         num_workers=32):
    """
    The main function
    """
    with open(config_fname, 'r') as file_handle:
        config = yaml.full_load(file_handle)

    print('\n================= Configuration =================')
    print_dict(config)
    print('=================================================\n')

    if input('Looks great?[Y/n]') == 'Y':
        PostProcess(config).process(num_workers)


if __name__ == '__main__':
    fire.Fire(main)
