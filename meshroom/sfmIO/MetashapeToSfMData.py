"""
Meshroom node for converting Agisoft Metashape XML to AliceVision SfMData.
"""

from meshroom.core import desc


class MetashapeToSfMData(desc.Node):
    """Convert Agisoft Metashape XML to AliceVision SfMData.

    Reads an Agisoft Metashape XML file and produces an AliceVision SfMData
    JSON file that can be used directly in a Meshroom pipeline.

    Coordinate conventions are handled automatically:
    - Metashape camera transforms (cam2world, row-major) are converted with the
      world-axis correction required by AliceVision/Meshroom.
    - Component (chunk) transforms are applied before the correction.
    - Sensor calibration (focal length, principal point, distortion) is
      converted from Metashape pixel units to AliceVision conventions.
    """

    category = "SfmIO"

    inputs = [
        desc.File(
            name="xmlPath",
            label="Metashape XML",
            description="Path to the Agisoft Metashape XML file.",
            value="",
        ),
        desc.File(
            name="imagesFolder",
            label="Images Folder",
            description=(
                "Optional folder containing the source images. "
                "When provided, image paths in the output SfMData point to this folder "
                "and the image extension is detected automatically."
            ),
            value="",
        ),
        desc.File(
            name="referenceSfmData",
            label="Reference SfMData",
            description=(
                "Optional path to a reference SfMData file. "
                "When provided, view IDs are matched by image name instead of being generated."
            ),
            value="",
        ),
        desc.FloatParam(
            name="sensorWidth",
            label="Sensor Width (mm)",
            description="Physical sensor width in millimetres used to convert the focal length.",
            value=36.0,
            range=(1.0, 100.0, 0.1),
        ),
        desc.FloatParam(
            name="sensorHeight",
            label="Sensor Height (mm)",
            description="Physical sensor height in millimetres.",
            value=24.0,
            range=(1.0, 100.0, 0.1),
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="SfMData",
            description="Path to the output AliceVision SfMData JSON file.",
            value="{nodeCacheFolder}/sfmData.json",
        ),
    ]

    def process(self, node):
        from pyalicevisionlib.scripts.metashape_to_sfmdata import convert_metashape_to_sfmdata

        images_folder = node.imagesFolder.value if node.imagesFolder.value else None
        reference = node.referenceSfmData.value if node.referenceSfmData.value else None

        convert_metashape_to_sfmdata(
            xml_path=node.xmlPath.value,
            output_path=node.output.value,
            sensor_width=node.sensorWidth.value,
            sensor_height=node.sensorHeight.value,
            images_folder=images_folder,
            reference_sfmdata=reference,
        )
