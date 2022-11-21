from typing import Union
from collections.abc import MutableMapping


class Tni:
    def __init__(self,
                 variable_type: str,
                 value_mapping: dict = None,
                 column_mapping: dict = None,
                 encoded_column_mapping: dict = None,
                 upper_bound=None,
                 lower_bound=None,
                 cutter: dict = None,
                 _categorical: dict = None):
        """
        :param variable_type: Data Types of each variable. Returned with Module 1
        :param value_mapping: Mapping of Imputation and Categorical Variables
        :param column_mapping: Describes the type of Transformation done to each Variable
        :param encoded_column_mapping: Bounds for Continuous-Binary Variables?
        :param upper_bound: Upper Capping Bounds of Continuous Variables
        :param lower_bound: Lower Capping Bounds of Continuous Variables
        :param cutter: List of cuts for the Continuous-Binary Variables?
        :param _categorical: Variables to be dropped after Categorical Transformations?
        :return: Data Set with TnI variables; Transformations and Imputations
        """

        # TODO reimplement dicts as TypedDict objects since the keys are statically defined.

        if variable_type not in ['numeric', 'categorical', 'binary', 'categorical+other',
                                 'continuous']:
            raise ValueError(f"variable type {variable_type} is not recognised")
        self.variable_type = variable_type
        # dict of args needed for pd cut
        self.cutter = cutter

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        # TODO coalesce into loop possibly for better aesthetics (maybe)
        # value mapping describes variable values that were changed as part of tni
        # {original value: new value}

        if not value_mapping:
            self.value_mapping = dict()
        else:
            self.value_mapping = value_mapping

        # Column mapping describes columns derived from the base variable column

        if not column_mapping:
            self.column_mapping = dict()
        else:
            self.column_mapping = column_mapping

        # encoded mappings show the new one hot encoded columns with its respective boundaries in pandas cut format
        # eg {one_hot_col_name : (500000, 500000000]

        if not encoded_column_mapping:
            self.encoded_column_mapping = dict()
        else:
            self.encoded_column_mapping = encoded_column_mapping

        # variables for categorical specific tni_information
        # {drop_cols: list of columns to be dropped after creating dummy variables}
        # TODO possibly switch to named tuple in class init
        if not _categorical:
            self._categorical = {'drop_columns': []}
        else:
            self._categorical = _categorical

    def show_all(self):
        return {'variable_type': self.variable_type,
                'value_mapping': self.value_mapping,
                'column_mapping': self.column_mapping,
                'encoded_column_mapping': self.encoded_column_mapping,
                '_categorical': self._categorical}


class Variable:
    def __init__(self,
                 name: str,
                 kind: str = None,
                 tni: Tni = None,
                 variable_selection_status: str = None):
        self.acceptable_types = {'continuous', 'exclude', 'categorical+other', 'categorical', 'target_variable',
                                 'binary'}

        self.name = name
        if kind:
            if kind not in self.acceptable_types:
                raise ValueError(f"variable {name} defined as a {kind} variable,"
                                 f"acceptable kinds are {self.acceptable_types}")
        self.kind = kind
        self.one_hot_encoded_columns = list()
        self._tni = tni
        self.variable_selection_status = variable_selection_status
        self.variable_status_mapping = {4: 'considered',
                                        1: 'excluded by user',
                                        2: 'zero variance',
                                        3: 'high correlation with other IV',
                                        5: 'selected'}
        self.variable_status_mapping_reversed = {v: k for k, v in self.variable_status_mapping.items()}

        self.metrics = {'corr_dv': None,
                        'corr_direction': None,
                        'univ_reg': None,
                        'cluster': None,
                        'cluster_correlation': None,
                        'lasso': None}

    def __bool__(self):
        return True

    def set_variable_metrics(self, metrics_dt: dict) -> None:
        """
        function for defining variable metrics after variable reduction
        :param metrics_dt: dt with keys {corr_direction, univ_reg, cluster, cluster_correlation, lasso,}
        :return: None
        """
        self.metrics = metrics_dt

    def set_variable_type(self, kind: str) -> None:
        if kind not in self.acceptable_types:
            raise ValueError(f"attempting to define {self.name} as a {kind} variable,"
                             f"acceptable kinds are {self.acceptable_types}")

    def set_one_hot_encoding(self, encoded_cols: Union[str, list]) -> None:
        if isinstance(encoded_cols, list):
            self.one_hot_encoded_columns += encoded_cols
        elif isinstance(encoded_cols, str):
            self.one_hot_encoded_columns.append(encoded_cols)
        else:
            raise TypeError(f"encoded_cols should be a list or string type"
                            f"instead it is a {type(encoded_cols)}")

    def get_or_create_tni(self):
        if not self._tni:
            # warnings.warn(f"tni processing not yet done for {self.name}")
            self._tni = Tni(variable_type=self.kind)
        return self._tni

    def tni(self):
        """
        alias for self.get_or_create_tni() for easier code formatting
        :return:
        """
        return self.get_or_create_tni()

    @property
    def processed_by_tni(self):
        if self._tni:
            return True
        else:
            return False

    def set_tni_value_mapping(self, value_mapping):
        self.get_or_create_tni().value_mapping = value_mapping

    def set_tni_column_mapping(self, column_mapping):
        self.get_or_create_tni().column_mapping = column_mapping

    def set_tni_encoded_column_mapping(self, encoded_column_mapping):
        self.get_or_create_tni().encoded_column_mapping = encoded_column_mapping

    def set_variable_selection_status(self, status_code: Union[str, int]) -> None:
        if isinstance(status_code, int):
            self.variable_selection_status = status_code
        elif status_code in self.variable_status_mapping_reversed:
            self.variable_selection_status = self.variable_status_mapping_reversed[status_code]
        else:
            raise ValueError(f"unrecognised status code {status_code}")

    def get_transformed_tni_variable_name(self) -> str:
        if self.kind in ('continuous', 'categorical'):
            if self.processed_by_tni:
                return self.get_or_create_tni().column_mapping['tni_processed']
        else:
            return self.name


class VariableCollection(MutableMapping):
    # TODO decide on going completely case agnostic
    def __init__(self):
        """
        custom dict style object with a hidden dictionary to account for different ways of referring to a variable
        Designed to be initiated as a blank dict type object with keys and values to be iteratively added
        """
        # TODO combine all dicts into single master dict
        # reverse mapping of tni_processed name to original variable name
        # e.g {estcash_tni_processed: estcash}
        # used to help lookup variable name using their transformed counterpart
        self.tni_reverse = dict()
        # lower cap version of tni_reverse.
        self.tni_reverse_lower = dict()

        # main underlying dictionary object for access.
        self._dt = dict()

        # lower case version of the underlying _dt. Helps correct against case issues.
        self._lower_map = {key.lower(): key for key in self._dt.keys()}

        # variable types used for actual modelling
        self.modelling_types = {'continuous', 'categorical', 'binary'}
        self._assigned_dict = dict()

    def show_types(self) -> dict:
        """
        function to show the variable type for all variables
        :return: dict
        """
        return {k: v.kind for k, v in self._dt.items()}

    def check_binary_columns(self,
                             item: str) -> Union[None, Variable]:
        """
        function that allow the use of binary variable name to refer to the underlying variable.
        e.g variable estcash, can be referred to using estcash_bi_(10990, 2000]
        :param item: name of binary variable to check against known variables
        :return: Variable, or None if there is no match.
        """
        # really works off the assumption that after the model init stage and tni stage, variable names and
        # binary columns are fixed and will not regenerate _assigned_dict mapping on every call
        if not self._assigned_dict:
            for k, v in self._dt.items():
                if v.kind in self.modelling_types:
                    for encoded in v.tni().encoded_column_mapping:
                        self._assigned_dict[encoded] = v

        if item in self._assigned_dict:
            return self._assigned_dict[item]
        else:
            return None

    # implementation of abstract methods
    def __getitem__(self, item: object) -> object:
        """
        custom implementation of baseline dictionary get,
        accounts for lower case searches, e.g user trying to get estcash, but calls self[EstCash]. Function will impute
        and pull correct variable underneath.

        Will do the same for tni_processed names. E.g self[estcash_tni_processed] will return the correct estcash
        variable object.

        Will do the same for binary varaible names. e.g self[estcash_bi_(512341, 5123]] will look at binary variables
        in each Variable object and attempt to match if possible

        :param item: object to search for in self._dt
        :return: matching object in _dt
        """
        # TODO add method for handling lookup of categorical binary variables.
        if item not in self._dt.keys():
            # check if lower in
            if item.lower() in self._lower_map:
                return self._dt[self._lower_map[item]]
            else:
                # check if tni_processed name matches
                if not self.tni_reverse:
                    self.refresh_tni_reverse()

                if item in self.tni_reverse:
                    return self._dt[self.tni_reverse[item]]

                elif self.check_binary_columns(item):
                    return self.check_binary_columns(item)
                # default option if nothing can be found. Might throw key error
                else:
                    return self._dt[self.tni_reverse_lower[item.lower()]]
        else:
            return self._dt[item]

    def __setitem__(self, key, value):
        """
        nothing fancy. standard implementation of setting a dictionary object. Interacts with _dt only
        :param key:
        :param value:
        :return:
        """
        self._dt[key] = value
        self._lower_map[key.lower()] = key

    def __delitem__(self, key):
        """
        standard implementation of delete item, set to work on _dt
        :param key:
        :return:
        """
        del self._dt[key]

    def __iter__(self):
        """
        custom implementation of iter method, this makes looping through the VariableCollection easier. e.g
        ```
        # assume VariableCollection is populated and defined as variables
        for i in variables:
            print(type(i))
            print(i.name)
            print(i.get_or_create_tni().show_all())
        ```
        This will show only Variable objects.

        Allows easy iteration through the underlying Variable objects.
        :return:
        """
        return iter(self._dt.values())

    def __len__(self) -> int:
        """
        counts number of variables existing.
        :return: number of variables
        """
        return len(self._dt)

    def items(self):
        return self._dt.items()

    def bool_get(self, item):
        """
        custom search, returns false if item cannot be found
        :param item:
        :return:
        """
        try:
            return self.__getitem__(item)
        except KeyError:
            return False

    def get_numeric_variables(self) -> set:
        return {k for k, v in self.items() if v.kind == 'continuous'}

    def get_categorical_variables(self) -> set:
        return {k for k, v in self.items() if v.kind == 'categorical'}

    def get_binary_variables(self) -> set:
        return {k for k, v in self.items() if v.kind == 'binary'}

    def get_excluded_variables(self) -> set:
        return {k for k, v in self.items() if v.kind not in {'binary', 'categorical', 'continuous'}}

    def get_active_variables_tni(self) -> set:
        # get all active variables with tni processed where possible
        res = self.get_numeric_variables()
        res.update(self.get_categorical_variables())

        tni_variables_to_add = []
        tni_variables_to_remove = []
        for var in res:
            if self._dt[var].processed_by_tni:
                for k, _tni_name in self._dt[var].get_or_create_tni().column_mapping.items():
                    if k != 'tni_assigned_bin':
                        tni_variables_to_add.append(_tni_name)
                        self.tni_reverse[_tni_name] = var
                        self.tni_reverse_lower[_tni_name.lower()] = var
                tni_variables_to_remove.append(var)

        res - set(tni_variables_to_remove)
        res.update(tni_variables_to_add)
        res.update(self.get_binary_variables())
        return res

    def get_active_variables(self) -> set:
        res = self.get_numeric_variables()
        res.update(self.get_categorical_variables())
        res.update(self.get_binary_variables())
        return res

    def refresh_tni_reverse(self):
        _ = self.get_active_variables_tni()

    def keys(self):
        return self._dt.keys()
