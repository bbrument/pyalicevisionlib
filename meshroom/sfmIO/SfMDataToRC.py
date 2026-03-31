"""
Meshroom node for converting AliceVision SfMData to RealityCapture XMP camera files.
"""

from meshroom.core import desc

from pyalicevisionlib.scripts.sfmdata_to_rc import convert_sfmdata_to_rc


class SfMDataToRC(desc.Node):
    """Convert AliceVision SfMData to RealityCapture XMP camera files.

    Reads an AliceVision SfMData file (.json / .sfm / .abc) and writes one XMP
    file per view into the specified output folder.  Images can optionally be
    copied alongside the XMP files so that the folder is immediately importable
    into RealityCapture.

    Coordinate conventions are handled automatically:
    - AV rotation (cam2world) is converted to RC rotation (world2cam, row-major).
    - AV principal point (pixel offset) is converted to RC normalised value.
    - World coordinate axes are corrected to match RealityCapture conventions.
    """

    category = "SfmIO"

    inputs = [
        desc.File(
            name="input",
            label="SfMData",
            description="Path to the input AliceVision SfMData file (.json, .sfm or .abc).",
            value="",
        ),
        desc.StringParam(
            name="imagesFolderName",
            label="Images Sub-Folder Name",
            description=(
                "Name of the sub-folder inside the output folder where images are copied. "
                "Only used when 'Copy Images' is enabled."
            ),
            value="images",
        ),
        desc.BoolParam(
            name="copyImages",
            label="Copy Images",
            description="Copy the source images into the output folder next to the XMP files.",
            value=True,
        ),
        desc.FloatParam(
            name="sensorWidth",
            label="Sensor Width (mm)",
            description=(
                "Default sensor width in millimetres used to convert focal length to the "
                "35 mm-equivalent format expected by RealityCapture. "
                "The value stored in the SfMData is used when available."
            ),
            value=36.0,
            range=(1.0, 100.0, 0.1),
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output Folder",
            description="Folder containing the generated XMP files (and optionally the copied images).",
            value="{nodeCacheFolder}/rc_export",
        ),
    ]

    def process(self, node):
        convert_sfmdata_to_rc(
            sfmdata_path=node.input.value,
            output_folder=node.output.value,
            images_folder_name=node.imagesFolderName.value,
            copy_images=node.copyImages.value,
            sensor_width=node.sensorWidth.value,
        )
