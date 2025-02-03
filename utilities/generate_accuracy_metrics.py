"""Contains functions for generating accuracy metrics."""

import copy
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


BLANK_CONF_MATRIX: Dict[str, int] = {
    "true_positive": 0,
    "false_positive": 0,
    "true_negative": 0,
    "false_negative": 0,
}


def read_ground_truth_data(filepath: Path) -> Dict[str, Dict]:
    """Reads the ground truth data into a dictionary.

    Args:
        `filepath` (Path):
            The filepath to the excel file with the ground truth data.

    Returns:
        A dictionary mapping the name of the chart to its ground truth data.
    """
    read_data_from_sheet = lambda sheet_name: pd.read_excel(
        str(filepath), sheet_name=sheet_name
    )
    wide_df_to_dictionary = lambda df: df.set_index("chart_id").T.to_dict()

    def blood_pressure_df_to_dictionary(bp_df):
        """Reads the blood pressure dataframe to a dictionary."""
        data: Dict = dict()
        unique_charts: List[str] = bp_df["chart_id"].unique().tolist()
        for chart_name in unique_charts:
            d = dict()
            for _, row in bp_df[bp_df["chart_id"] == chart_name].iterrows():
                d[row.timestamp] = {
                    "systolic": row.systolic,
                    "diastolic": row.diastolic,
                    "heart_rate": row.heart_rate,
                }
            data[chart_name] = d
        return data

    def intraop_binned_numerical_time_to_dictionary(ibnt_df):
        """Reads the digit data from the dataframes."""
        data = dict()
        for chart_name in ibnt_df["chart_id"].unique().tolist():
            df_slice = ibnt_df[ibnt_df["chart_id"] == chart_name].copy()
            chart_data = dict()
            for physio_indicator in df_slice["name"].unique().tolist():
                physio_df = df_slice[df_slice["name"] == physio_indicator].copy()
                physio_data = dict()
                for ix, row in physio_df.iterrows():
                    physio_data.update({row.timestamp: row.value})
                chart_data[physio_indicator] = physio_data
            data[chart_name] = chart_data
        return data

    wide_dataframes: List[pd.DataFrame] = [
        read_data_from_sheet("Intraop_Binned_Numerical"),
        read_data_from_sheet("Preop_Binned_Numerical"),
        read_data_from_sheet("Intraop_Checkboxes"),
        read_data_from_sheet("Preop_Checkboxes"),
    ]
    long_dataframes: List[pd.DataFrame] = [read_data_from_sheet("Multibox_Numerical")]
    data: Dict = dict()
    for key in wide_df_to_dictionary(wide_dataframes[0]).keys():
        data[key] = dict()

    for df in wide_dataframes:
        df_dict: Dict[str, Dict] = wide_df_to_dictionary(df)
        for key, value in df_dict.items():
            data[key].update(value)

    bp_and_hr_dict: Dict = blood_pressure_df_to_dictionary(
        read_data_from_sheet("Blood_Pressure_and_Heart_Rate")
    )
    for key, value in bp_and_hr_dict.items():
        data[key]["bp_and_hr"] = value

    intraop_binned_dict: Dict = intraop_binned_numerical_time_to_dictionary(
        read_data_from_sheet("Intraop_Binned_Numerical_Time")
    )
    for key, value in intraop_binned_dict.items():
        data[key]["physiological_indicators"] = value

    return data


def compute_bp_and_hr_accuracy(
    extracted_data: Dict[str, Dict],
    ground_truth_data: Dict[str, Dict],
) -> float:
    """Computes the surgical and anesthesia timing accuracy.

    Only computes accuracy for systolic, diastolic, and heart rate values that
    appear in both the ground truth and predicted data.

    Args:
        `extracted_data` (Dict[str, Dict]):
            A dictionary mapping the name of charts to the output dictionaries.
        `ground_truth_data` (Dict[str, Dict]):
            The ground truth data.

    Returns:
        The proportion of correct charts divided by all the charts.
    """
    accuracy: Dict[str, float] = {"systolic": [], "diastolic": [], "heart_rate": []}
    for chart_name in list(extracted_data.keys()):
        if any(
            [
                chart_name not in ground_truth_data.keys(),
                extracted_data[chart_name].get("bp_and_hr") is None,
                ground_truth_data[chart_name].get("bp_and_hr") is None,
            ]
        ):
            continue
        predicted_bp: Dict = extracted_data[chart_name]["bp_and_hr"]
        true_bp: Dict = ground_truth_data[chart_name]["bp_and_hr"]
        timestamps = sorted(list(true_bp.keys()))
        for timestamp in timestamps:
            if timestamp not in predicted_bp.keys():
                continue
            for dtype in ["systolic", "diastolic", "heart_rate"]:
                predicted_value: Optional[str] = predicted_bp[timestamp].get(dtype)
                true_value: Optional[str] = true_bp[timestamp].get(dtype)
                try:
                    accuracy[dtype].append(
                        int(predicted_value.split("_")[0])
                        - int(true_value.split("_")[0])
                    )
                except:
                    pass
    return accuracy


def compute_bp_and_hr_confusion_matrix(
    extracted_data: Dict[str, Dict],
    ground_truth_data: Dict[str, Dict],
) -> Dict[str, Dict[str, int]]:
    """Computes the detection accuracy confusion matrix for the blood pressure.

    Args:
        `extracted_data` (Dict[str, Dict]):
            A dictionary mapping the name of charts to the output dictionaries.
        `ground_truth_data` (Dict[str, Dict]):
            The ground truth data.

    Returns:
        A dictionary of three confusion matrices, one for systolic, diastolic, and heart rate.
    """

    confusion_matrices: Dict[str, Dict[str, int]] = {
        "systolic": copy.deepcopy(BLANK_CONF_MATRIX),
        "diastolic": copy.deepcopy(BLANK_CONF_MATRIX),
        "heart_rate": copy.deepcopy(BLANK_CONF_MATRIX),
    }

    for chart_name in list(extracted_data.keys()):
        if any(
            [
                chart_name not in ground_truth_data.keys(),
                extracted_data[chart_name].get("bp_and_hr") is None,
                ground_truth_data[chart_name].get("bp_and_hr") is None,
            ]
        ):
            continue
        predicted_bp: Dict = extracted_data[chart_name]["bp_and_hr"]
        true_bp: Dict = ground_truth_data[chart_name]["bp_and_hr"]
        timestamps = sorted(list(true_bp.keys()))
        for timestamp in timestamps:
            if timestamp not in predicted_bp.keys():
                for dtype in ["systolic", "diastolic", "heart_rate"]:
                    confusion_matrices[dtype]["false_negative"] += 1
                continue
            for dtype in ["systolic", "diastolic", "heart_rate"]:
                predicted_value: Optional[str] = predicted_bp[timestamp].get(dtype)
                true_value: Optional[str] = true_bp[timestamp].get(dtype)
                if predicted_value is not None and true_value is not None:
                    confusion_matrices[dtype]["true_positive"] += 1
                elif predicted_value is None and true_value is not None:
                    confusion_matrices[dtype]["false_negative"] += 1
                elif predicted_value is not None and true_value is None:
                    confusion_matrices[dtype]["false_positive"] += 1
                elif predicted_value is None and true_value is None:
                    confusion_matrices[dtype]["true_negative"] += 1
                else:
                    pass
    return confusion_matrices


def compute_physiological_indicator_accuracy(
    extracted_data,
    ground_truth_data,
):
    """Computes the inference accuracy of the physiological indicator extraction.

    Args:
        `extracted_data` (Dict[str, Dict]):
            A dictionary mapping the name of charts to the output dictionaries.
        `ground_truth_data` (Dict[str, Dict]):
            The ground truth data.

    Returns:
        A dictionary of absolute differences between ground truth and extract values indexed by
        the name of the physiological indicator.
    """
    accuracy: Dict[str, float] = {
        "inhaled_volatile": [],
        "spo2": [],
        "etco2": [],
        "fio2": [],
        "temperature": [],
        "tidal_volume": [],
        "respiratory_rate": [],
        "urine_output": [],
        "blood_loss": [],
    }

    for chart_name in list(extracted_data.keys()):
        if any(
            [
                chart_name not in ground_truth_data.keys(),
                extracted_data[chart_name].get("physiological_indicators") is None,
                ground_truth_data[chart_name].get("physiological_indicators") is None,
            ]
        ):
            continue
        predicted_physio: Dict = extracted_data[chart_name]["physiological_indicators"]
        true_physio: Dict = ground_truth_data[chart_name]["physiological_indicators"]
        indicators = sorted(list(true_physio.keys()))
        for indicator in indicators:
            if indicator not in predicted_physio.keys():
                continue
            for timestamp in predicted_physio[indicator].keys():
                predicted_value: Optional[str] = predicted_physio[indicator][timestamp]
                true_value: Optional[str] = true_physio[indicator][timestamp]
                accuracy[indicator].append(abs(true_value - float(predicted_value)))
    return accuracy


def compute_physiological_indicator_conf_matrix(
    extracted_data,
    ground_truth_data,
):
    """Computes the detection accuracy of the physiological indicators.

    Args:
        `extracted_data` (Dict[str, Dict]):
            A dictionary mapping the name of charts to the output dictionaries.
        `ground_truth_data` (Dict[str, Dict]):
            The ground truth data.

    Returns:
        A dictionary of confusion matrices indexed by the name of the physiological indicator.
    """
    confusion_matrices: Dict[str, float] = {
        "inhaled_volatile": copy.deepcopy(BLANK_CONF_MATRIX),
        "spo2": copy.deepcopy(BLANK_CONF_MATRIX),
        "etco2": copy.deepcopy(BLANK_CONF_MATRIX),
        "fio2": copy.deepcopy(BLANK_CONF_MATRIX),
        "temperature": copy.deepcopy(BLANK_CONF_MATRIX),
        "tidal_volume": copy.deepcopy(BLANK_CONF_MATRIX),
        "respiratory_rate": copy.deepcopy(BLANK_CONF_MATRIX),
        "urine_output": copy.deepcopy(BLANK_CONF_MATRIX),
        "blood_loss": copy.deepcopy(BLANK_CONF_MATRIX),
    }
    for chart_name in list(extracted_data.keys()):
        if any(
            [
                chart_name not in ground_truth_data.keys(),
                extracted_data[chart_name].get("physiological_indicators") is None,
                ground_truth_data[chart_name].get("physiological_indicators") is None,
            ]
        ):
            continue
        predicted_physio: Dict = extracted_data[chart_name]["physiological_indicators"]
        true_physio: Dict = ground_truth_data[chart_name]["physiological_indicators"]
        indicators = sorted(list(true_physio.keys()))
        for indicator in indicators:
            if indicator not in predicted_physio.keys():
                confusion_matrices[indicator]["false_negative"] += len(
                    true_physio[indicator]
                )
                continue
            for timestamp in predicted_physio[indicator].keys():
                predicted_value: Optional[str] = predicted_physio[indicator].get(
                    timestamp
                )
                true_value: Optional[str] = true_physio[indicator].get(timestamp)
                if predicted_value is not None and true_value is not None:
                    confusion_matrices[indicator]["true_positive"] += 1
                elif predicted_value is None and true_value is not None:
                    confusion_matrices[indicator]["false_negative"] += 1
                elif predicted_value is not None and true_value is None:
                    confusion_matrices[indicator]["false_positive"] += 1
                elif predicted_value is None and true_value is None:
                    confusion_matrices[indicator]["true_negative"] += 1
                else:
                    pass
    return confusion_matrices


def compute_checkbox_confusion_matrices(
    extracted_data,
    ground_truth_data,
):
    """Computes both detection and inference statistics for checkboxes.

    Args:
        `extracted_data` (Dict[str, Dict]):
            A dictionary mapping the name of charts to the output dictionaries.
        `ground_truth_data` (Dict[str, Dict]):
            The ground truth data.

    Returns:
        A confusion matrix with an additional 'missing' category that encodes detections
        that were completely missed by the model.
    """
    conf_matrix = copy.deepcopy(BLANK_CONF_MATRIX)
    conf_matrix["missing"] = 0
    for chart_name in extracted_data.keys():
        true_checkboxes = {
            k: v
            for (k, v) in ground_truth_data[chart_name].items()
            if v in ["checked", "unchecked"]
        }
        extracted_checkboxes = dict()
        extracted_checkboxes.update(
            extracted_data[chart_name]["intraoperative_checkboxes"]
        )
        extracted_checkboxes.update(
            extracted_data[chart_name]["preoperative_checkboxes"]
        )
        for key in extracted_data[chart_name].keys():
            if extracted_data[chart_name][key] in ["checked", "unchecked"]:
                extracted_checkboxes[key] = extracted_data[chart_name][key]
        for key in true_checkboxes.keys():
            if extracted_checkboxes.get(key) == None:
                print(key)
                conf_matrix["missing"] += 1
                continue
            if (
                true_checkboxes[key] == "checked"
                and extracted_checkboxes[key] == "checked"
            ):
                conf_matrix["true_positive"] += 1
            elif (
                true_checkboxes[key] == "unchecked"
                and extracted_checkboxes[key] == "checked"
            ):
                conf_matrix["false_positive"] += 1
            elif (
                true_checkboxes[key] == "checked"
                and extracted_checkboxes[key] == "unchecked"
            ):
                conf_matrix["false_negative"] += 1
            elif (
                true_checkboxes[key] == "unchecked"
                and extracted_checkboxes[key] == "unchecked"
            ):
                conf_matrix["true_negative"] += 1
    return conf_matrix


def compute_lab_value_accuracy(
    extracted_data,
    ground_truth_data,
) -> Dict[str, Dict[str, int]]:
    """Computes the accuracy of the extracted lab values.

    Args:
        `extracted_data` (Dict[str, Dict]):
            A dictionary mapping the name of charts to the output dictionaries.
        `ground_truth_data` (Dict[str, Dict]):
            The ground truth data.

    Returns:
        A dictionary of dictionaries, each containing a "correct" and "total" amount of
        it's indexing string's values in the ground truth dataset.
        Ex: {"hgb": {"correct": 25, "total": 30}}.
    """

    def is_nan(value):
        if isinstance(value, str):
            return True
        elif value is None:
            return True
        else:
            return np.isnan(value)

    lab_val_keys = [
        "hgb",
        "hct",
        "plt",
        "na",
        "k",
        "cl",
        "urea",
        "creatinine",
        "ca",
        "mg",
        "po4",
    ]
    blank = {"correct": 0, "total": 0}
    accuracies = {lv: copy.deepcopy(blank) for lv in lab_val_keys}
    for chart_name in extracted_data.keys():
        for lv in lab_val_keys:
            if is_nan(ground_truth_data[chart_name].get(lv)):
                continue

            if (
                float(extracted_data[chart_name]["lab_values"][lv].replace(" ", ""))
                == ground_truth_data[chart_name][lv]
            ):
                accuracies[lv]["correct"] += 1
            accuracies[lv]["total"] += 1
    return accuracies
