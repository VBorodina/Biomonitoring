import numpy as np
import pandas as pd
import glob
import os.path
from copy import deepcopy
#ELELE

class Experiment:
    """
    A class for creating an Experiment object with preprocessed datasets containing offline(HPLC, bio dry mass), online (MFCS), CO2(Bluesense) data. The class assumes that there is a common path where the whole data is present within. 

    :param path: Path of overarching directory, the metadata.xlsx file and the directories containing the data (e.g. online, offline and CO2) must be within this directory.
    :type path: str
    :param exp_nr: Number of Experiment in the filename (e.g. offline_3). Number in filename and in metadata must match and should correspond to the same experiment.
    :type exp_nr: int
    :param meta_path: metadata file name within path.
    :type meta_path: str
    :param types: List with measurement data types.
    :type types: list with strings
    :param prefix: List with file prefix names within path, must match with types.
    :type prefix: list with strings
    :param suffix: List with data endings (data types), must match with types
    :type suffix: list with strings
    :param index_ts: List with position of the index column of the raw data.
    :type index_ts: list with int's
    :param read_csv_settings: List with pd.read_csv settings to read the correspondig raw data to pd.DataFrame.
    :type read_csv_settings: list with dicts
    :param to_datetime_settings: List with settings to convert Timestamp column in raw data to pd.Timestamp
    :return: Experiment object contaning the measurement data.

        """   


    def __repr__(self):
        """Representation of Experiment object in the print statement"""
        return  """Experiment(\"{path}\" , \"{exp_nr}\")""".format(path= self._path, exp_nr= self.exp_nr)

    #format_ts kommt raus wenn du eh to_datetime_settings_gibst

    def __init__(self, path, exp_nr, meta_path = "metadata.xlsx", types= ["off", "on", "CO2"]
    , prefix = ["offline/offline_", "online/online_", "CO2/CO2_"]
    , suffix = [".csv", ".CSV", ".dat"]
    , index_ts = [0,0,0]
    , read_csv_settings = [     dict(sep=";", encoding= 'unicode_escape', header = 0, usecols = None)
    , dict(sep=";",encoding= "unicode_escape",decimal=",", skiprows=[1,2] , skipfooter=1, usecols = None, engine="python")
    , dict(sep=";", encoding= "unicode_escape", header = 0, skiprows=[0], usecols=[0,2,4], names =["ts","CO2", "p"])   ]
    , to_datetime_settings = [  dict(format = "%d.%m.%Y %H:%M", exact= False, errors = "coerce")
    , dict(format = "%d.%m.%Y  %H:%M:%S", exact= False, errors = "coerce")
    , dict(format = "%d.%m.%Y %H:%M:%S", exact= False, errors = "coerce")     ]

    ):

    # , filtering_columns = False, filter_on = ["base_rate"], filter_off = ["cX", "cS", "cE"], filter_CO2 = ["CO2"]
        pd.options.mode.chained_assignment = None       #because of pandas anoying error
        assert type(path) is str, "The given Path must be of type str"
        assert type(meta_path) is str, "The given meta_path must be of type str"
        assert all(isinstance(i, list) for i in (types, prefix, suffix, index_ts, read_csv_settings, to_datetime_settings)), "The arguments types, prefix, suffix, index_ts, read_csv_settings, to_datetime_settings have to be of type list and must match"
        assert all(len(types) == len(i) for i in (prefix, suffix, index_ts, read_csv_settings, to_datetime_settings)), "The length of types must match with format_ts, prefix, suffix, index_ts, read_csv_settings here {0} does not match".format(i)
        

        info = dict.fromkeys(types)
        types_dict = dict(enumerate(types))         #create information dict of dicts
        for i, typ in types_dict.items():
            info[typ] = dict.fromkeys(["format_ts", "prefix", "suffix", "index_ts", "read_csv_settings", "to_datetime_settings"])
            info[typ]["prefix"] = prefix[i]
            info[typ]["suffix"] = suffix[i]
            info[typ]["index_ts"] = index_ts[i]
            info[typ]["read_csv_settings"] = read_csv_settings[i]
            info[typ]["to_datetime_settings"] = to_datetime_settings[i]
              
        self._path = path 
        self.exp_nr = exp_nr

        file_path = {}
        for typ in types:
            file_path[typ] = os.path.join(path, info[typ]["prefix"] + str(exp_nr) + info[typ]["suffix"])

        

        self.dataset = {}
        for typ, p in file_path.items():
            self.dataset[typ] = self.read_data(path = p, index_ts = info[typ]["index_ts"], read_csv_settings = info[typ]["read_csv_settings"], to_datetime_settings = info[typ]["to_datetime_settings"])


        metadata_all = pd.read_excel(os.path.join(path, meta_path), index_col = 0)
        assert exp_nr in metadata_all.index.values, "Experiment have to be in metadata"
        self.metadata = metadata_all.loc[exp_nr]

        # filter for relevant time interval
        start = self.metadata["start"]
        end = self.metadata["end1"]

        
        for dskey in self.dataset.keys():
            self.time_filter(dskey, start, end)

        for dskey in self.dataset.keys():
            self._calc_t(dskey = dskey, start = start)
        
        self.calc_rate("on", "BASET")
        
        
    
    def time_filter(self, dskey, start = None, end = None):
        """Function to filter according to process time  

        :param dskey: Dataset key correspond to one of the measuring types (e.g. "off" , "CO2").
        :type dskey: dict key
        :param start: Timestamp of start point.
        :type start: pd.Timestamp or str
        :param end:  Timestamp of end point.
        :type end: pd.Timestamp or str
        :return: Filtered dataframes in dataset.

        """
        df = self.dataset[dskey]        # df = deepcopy(self.dataset[key] to avoid error message?) or pd.options.mode.chained_assignment = None in constructor

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        #case distinction depening on given start and/or end time points
        if start is None:

            if end is None:
                pass
            
            if end is not None:
                df = df[(df["ts"] <= end)]  

        elif start is not None:

            if end is None:
                df = df[(df["ts"] >= start)]  
            
            if end is not None: 
                df = df[(df["ts"] >= start ) &  (df["ts"] <= end)] 

        self.dataset[dskey] = df



    def calc_rate(self, dskey, col):
        """ Function to calculate the time derivative of a variable with finite differences.

        :param dskey: Dataset key correspond to one of the measuring types (e.g. "off" , "CO2")
        :type dskey: dict key
        :param col: Column in dataframe = variable for calculating the rate
        :type col: str
        :return: New column in dataframe named col_rate = time derivative of col.

        """

        df = self.dataset[dskey]

        try:
            df[col + "_rate"] = df[col].diff() / np.diff(df.index, prepend= 1)
        
        except:
            df[col] = pd.to_numeric(df[col] , downcast="float" , errors="coerce") # some values in BASET were recognized as string
            df[col + "_rate"] = df[col].diff() / np.diff(df.index, prepend= 1)

        self.dataset[dskey] = df

    def _calc_t(self, dskey, start = None):
        """ Calculates process time as decimal number.

        :param dskey: Dataset key correspond to one of the measuring types (e.g. "off" , "CO2")
        :type dskey: dict key
        :param start: Timestamp of start point
        :type start: pd.Timestamp or str
        :return: Time as decimal number in a new column "t"
        
        """

        df = self.dataset[dskey]

        if start is None:
            df["t"] = (df["ts"] - df["ts"][0]) / pd.Timedelta(1,"h")
        if start is not None:
            df["t"] = (df["ts"] - start) / pd.Timedelta(1,"h")

    
        df.set_index("t", inplace= True, drop= True)        #set t to index of dataframe
        
        self.dataset[dskey] = df

        
    def read_data(self, path, index_ts, read_csv_settings, to_datetime_settings):
        """ Function to read the measurement data with the corresponding settings

        :param path: Path of the measurement data.
        :type path: str 
        :param index_ts: Index of the timestamp column
        :type index_ts: int
        :param read_csv_settings: Pandas read_csv setings for this type of data.
        :type read_csv_settings: dict
        :param to_datetime_settings: Pandas to_datetime settings to convert timestamp column to pd.Timestamp
        :type to_datetime_settings: dict

        """

        df = pd.read_csv(path, **read_csv_settings)
        df["ts"] = pd.to_datetime(df.iloc[:, index_ts], **to_datetime_settings)
        return df


    def pop_dataframe(self, types):
        """Function to delete whole dataframes from the dataset, either of one or several types.

            :param types: Type of the measurement data.
            :type types: str or list of str's
            :return: Dataset without specific selected dataframe/s.
        """

        if type(types) is list:
            for typ in types:
                if typ in self.dataset.keys():
                    self.dataset.pop(typ)
                    
                else: 
                    raise ValueError("All types must be in dataset.keys() = ['on', 'off', 'CO2'] for G015")

        elif type(types) is str:
            if types in self.dataset.keys():
                self.dataset.pop(types)

            else:
                raise ValueError("Typ must be in dataset.keys() = e.g. ['on', 'off', 'CO2'] for G015")
        else:
            raise TypeError("Type of types must be a str or a list of strings")


