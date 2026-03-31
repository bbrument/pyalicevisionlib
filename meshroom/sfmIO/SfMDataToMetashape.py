"""
Meshroom node for converting AliceVision SfMData to Agisoft Metashape XML.
"""

from meshroom.core import desc

from pyalicevisionlib.scripts.sfmdata_to_metashape import convert_sfmdata_to_metashape


class SfMDataToMetashape(desc.Node):
    """Convert AliceVision SfMData to Agisoft Metashape XML.

    Reads an AliceVision SfMData file (.json / .sfm / .abc) and writes a
    Metashape-compatible XML file that can be imported directly into Agisoft
    Metashape.

    Coordinate conventions are handled automatically:
    - AV rotation (cam2world) is converted with the world-axis correction
      required by Metashape.
    - Focal length (mm) is converted to pixels using the sensor width.
    - Principal point and distortion parameters are passed through unchanged.
    """

    category = "SfmIO"

    inputs = [
        desc.File(
            name="input",
            label="SfMData",
            description="Path to the input AliceVision SfMData file (.json, .sfm or .abc).",
            value="",
        ),
        desc.FloatParam(
            name="sensorWidth",
            label="Sensor Width (mm)",
            description=(
                "Default sensor width in millimetres used to convert focal length to pixels. "
                "The value stored in the SfMData is used when available."
            ),
            value=36.0,
            range=(1.0, 100.0, 0.1),
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Metashape XML",
            description="Path to the output Agisoft Metashape XML file.",
            value="{nodeCacheFolder}/metashape.xml",
        ),
    ]

    def process(self, node):
        convert_sfmdata_to_metashape(
            sfmdata_path=node.input.value,
            output_path=node.output.value,
            sensor_width=node.sensorWidth.value,
        )
