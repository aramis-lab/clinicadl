import re
import time
from datetime import datetime, timedelta

from clinicadl.io.maps.summary import MapsSummary
from clinicadl.io.maps.training import TrainingSummary


def normalize_file(path):
    normalized = []
    with open(path) as f:
        for line in f:
            if re.match(r"\s*Date:", line.strip()):
                normalized.append(re.sub(r"Date: .*", "Date: <ignore>", line))
            elif re.match(r"\s*Path:", line.strip()):
                normalized.append(re.sub(r"Path: .*", "Path: <ignore>", line))
            else:
                normalized.append(line)
    return normalized


def test_summary(tmp_path):
    path = tmp_path / "summary.log"
    summary = MapsSummary(path)

    summary.create()

    summary.add_training_split(1)
    summary.add_training_split(2)
    summary.add_prediction_group("abc")
    summary.add_training_split(3)
    summary.add_prediction_group("efg")
    summary.add_test_group("abc")
    summary.add_test_group("efg")
    time.sleep(1)
    summary.add_training_split(1)
    summary.add_prediction_group("abc")
    summary.add_test_group("abc")

    actual_lines = normalize_file(path)

    expected_lines = [
        "==================== MAPS summary ====================\n",
        "\n",
        "Date: <ignore>\n",
        "Path: <ignore>\n",
        "\n",
        "---------------------- Training ----------------------\n",
        "\n",
        "Splits\n",
        "   - 1\n",
        "      Date: <ignore>\n",
        "   - 2\n",
        "      Date: <ignore>\n",
        "   - 3\n",
        "      Date: <ignore>\n",
        "\n",
        "--------------------- Prediction ---------------------\n",
        "\n",
        "Groups\n",
        "   - abc\n",
        "   - efg\n",
        "\n",
        "------------------------ Test ------------------------\n",
        "\n",
        "Groups\n",
        "   - abc\n",
        "   - efg\n",
        "\n",
    ]

    assert actual_lines == expected_lines

    # dates
    with open(path) as f:
        content = f.read()

    pattern = r"\d{4} [A-Za-z]{3} \d{2}, \d{2}:\d{2}:\d{2}"
    dates = [
        datetime.strptime(s, "%Y %b %d, %H:%M:%S") for s in re.findall(pattern, content)
    ]
    assert dates[1] >= (dates[0] + timedelta(seconds=1))
    assert dates[2] == dates[0]


def test_training_summary(tmp_path):
    path = tmp_path / "summary.log"
    summary = TrainingSummary(path)
    summary.create()
    summary.add_data_info(n_train_samples=1e6, n_val_samples=1e5)
    summary.add_training_end_info(n_epochs=1e3, interrupted=False)
    summary.add_info("\nabc")

    actual_lines = normalize_file(path)
    expected_lines = [
        "================================================ Training ================================================\n",
        "\n",
        "Date: <ignore>\n",
        "\n",
        "Trained with 1,000,000 samples\n",
        "Validated on 100,000 samples\n",
        "\n",
        "Training completed after 1,000 epochs\n",
        "\n",
        "abc",
    ]
    assert actual_lines == expected_lines

    summary.create()
    summary.add_training_end_info(n_epochs=1e3, interrupted=True)
    actual_lines = normalize_file(path)
    assert actual_lines[4] == "Training interrupted after 1,000 epochs\n"
