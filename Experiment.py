import numpy as np
import pandas as pd
import glob
import os.path
from copy import deepcopy

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
    :param format_ts: List with time stamp formats, must match with types.
    :type format_ts: list with strings
    :param prefix: List with file prefix names within path, must match with types.
    :type prefix: list with strings
    :param suffix: List with data endings (data types), must match with types
    :type suffix: list with strings

    :return: Experiment object contaning the measurement data.

        """   




    def __repr__(self):
        """Representation of Experiment object in the print statement"""
        return  """Experiment(\"{path}\" , \"{exp_nr}\")""".format(path= self._path, exp_nr= self.exp_nr)


    def __init__(self, path, exp_nr, meta_path = "metadata.xlsx", types= ["off", "on", "CO2"]
    , format_ts = ["%d.%m.%Y %H:%M", "%d.%m.%Y  %H:%M:%S", "%d.%m.%Y %H:%M:%S"]
    , prefix = ["offline/offline_", "online/online_", "CO2/CO2_"]
    , suffix = [".csv", ".CSV", ".dat"]):

    # , filtering_columns = False, filter_on = ["base_rate"], filter_off = ["cX", "cS", "cE"], filter_CO2 = ["CO2"]
    #read_csv_kwargs, to_datetime_kwargs keyword arguments für generische read_data funktion ?
    #read_csv_kwargs = [dict(sep=";", encoding= 'unicode_escape', header = 0, usecols = None), dict(sep=";",encoding= "unicode_escape",decimal=",", skiprows=[1,2] , skipfooter=1, usecols = None, engine="python"), dict(sep=";", encoding= "unicode_escape", header = 0, skiprows=[0], usecols=[0,2,4], names =["ts","CO2", "p"])]
        #wenn types format, prefix und suffix keine list dann mach list daraus

        pd.options.mode.chained_assignment = None       #because of pandas anoying error
        assert type(path) is str, "The given Path must be of type str"
        assert type(meta_path) is str, "The given meta_path must be of type str"
        for i in (format_ts, prefix, suffix):
            for j in i:
                assert isinstance(j, str), "Every element in format_ts, prefix and suffix must be of type str the element {0} in {1} is not a str".format(j, i)
        assert all(len(types) == len(i) for i in (format_ts, prefix, suffix)), "The length of types must match with format_ts, prefix and suffix, here {0} does not match".format(i)
        

        info = dict.fromkeys(types)
        types_dict = dict(enumerate(types))
        for i, typ in types_dict.items():
            info[typ] = dict.fromkeys(["format_ts", "prefix", "suffix"])
            info[typ]["format_ts"] = format_ts[i]
            info[typ]["prefix"] = prefix[i]
            info[typ]["suffix"] = suffix[i]

        
        self._path = path 
        self.exp_nr = exp_nr

        file_path = {}
        for typ in types:
            file_path[typ] = os.path.join(path, info[typ]["prefix"] + str(exp_nr) + info[typ]["suffix"])

        

        self.dataset = {}
        for typ, p in file_path.items():
            self.dataset[typ] = self.read_data(typ = typ, path = p, format_ts = info[typ]["format_ts"])


        self.wf = {dskey: 1.0 for dskey in self.dataset.keys()} #dict with weighting factors, 1 for default value

        
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

        

    def read_data(self, typ, path, format_ts):
        """ Function which decides how to read the measurement data, this function calls specific subfunctions to read specific types of measurement data
        
        :param typ: Type of the measurement data
        :type typ: str
        :param path: path of the measurement data.
        :type path: str
        :param format_ts: timestamp format in the raw data
        :type format_ts: str
        :returns: Pandas dataframe with measurement data and a timestamp column containing pd.Timestamps.
        
        """
        

        if typ == "off":
            df = self._read_offline(path, format_ts)

        elif typ == "on":
            df = self._read_online(path, format_ts)

        elif typ == "CO2":
            df = self._read_CO2(path, format_ts)
        
        return df

    def _read_offline(self, path, format_ts):
        """Special function to read offline data and convert string column to pd.Timestamp column"""
        df = pd.read_csv(path,sep=";", encoding= 'unicode_escape', header = 0, usecols = None)
        df["ts"] = pd.to_datetime(df.iloc[:, 0], format = format_ts, exact= False, errors = "coerce")
        #df["ts"] = pd.to_datetime(df.iloc[:, 0], format = format_ts) #old
        return df

    def _read_online(self, path, format_ts):
        """Special function to read online data and convert string column to pd.Timestamp column"""

        df = pd.read_csv(path ,sep=";",encoding= "unicode_escape",decimal=",", skiprows=[1,2] , skipfooter=1, usecols = None, engine="python")
        df["ts"] = pd.to_datetime(df.iloc[:, 0], format = format_ts, exact= False, errors = "coerce")   #new format but only necessary for CO2 data
        #df["ts"] = pd.to_datetime(df.iloc[:, 0], format = format_ts) #old
        return df
    
    def _read_CO2(self, path, format_ts):
        """Special function to read CO2 data and convert string column to pd.Timestamp column"""
        df = pd.read_csv(path, sep=";", encoding= "unicode_escape", header = 0, skiprows=[0], usecols=[0,2,4], names =["ts","CO2", "p"])
        df["ts"] = pd.to_datetime(df["ts"], format = format_ts, exact= False, errors = "coerce")
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
                    self.wf.pop(typ)
                else: 
                    raise ValueError("All types must be in dataset.keys() = ['on', 'off', 'CO2'] for G015")

        elif type(types) is str:
            if types in self.dataset.keys():
                self.dataset.pop(types)
                self.wf.pop(types)
            else:
                raise ValueError("Typ must be in dataset.keys() = e.g. ['on', 'off', 'CO2'] for G015")
        else:
            raise TypeError("Type of types must be a str or a list of strings")


